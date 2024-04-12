#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import model
from detection_layers.modules import MultiBoxLoss
from dataset import DeepfakeDataset
from lib.util import load_config, update_learning_rate, my_collate
import os 
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_train.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.', default=None)
    parser.add_argument('--local-rank', type=int, help='The local rank of GPU', default=None)
    parser.add_argument('--sbi', type=int, help='Experimental settings for SBI. \
                                                1) Swap-After-SBI \
                                                2) SBI-After-Swap', default=0)
    args = parser.parse_args()
    return args


def save_checkpoint(net, opt, save_path, epoch_num):
    os.makedirs(save_path, exist_ok=True)
    module = net.module
    model_state_dict = OrderedDict()
    for k, v in module.state_dict().items():
        model_state_dict[k] = torch.tensor(v, device="cpu")

    opt_state_dict = {}
    opt_state_dict['param_groups'] = opt.state_dict()['param_groups']
    opt_state_dict['state'] = OrderedDict()
    for k, v in opt.state_dict()['state'].items():
        opt_state_dict['state'][k] = {}
        opt_state_dict['state'][k]['step'] = v['step']
        if 'exp_avg' in v:
            opt_state_dict['state'][k]['exp_avg'] = torch.tensor(v['exp_avg'], device="cpu")
        if 'exp_avg_sq' in v:
            opt_state_dict['state'][k]['exp_avg_sq'] = torch.tensor(v['exp_avg_sq'], device="cpu")

    checkpoint = {
        'network': model_state_dict,
        'opt_state': opt_state_dict,
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}.pkl')


def load_checkpoint(ckpt, net, opt, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    # TODO: FIX module naming convention
    for k, v in checkpoint['network'] .items():
        name = k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    opt.load_state_dict(checkpoint['opt_state'])
    base_epoch = int(checkpoint['epoch']) + 1
    return net, opt, base_epoch


def train():
    args = args_func()

    dist_url = 'env://'
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.distributed.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(f'RANK {rank} WORLD SIZE {world_size} LOCAL_RANK {local_rank}')

    # load conifigs
    cfg = load_config(args.cfg)

    # init model.
    net = model.get(backbone=cfg['model']['backbone'])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = local_rank
    torch.cuda.set_device(local_rank)
    net = net.cuda(local_rank)
    # net = nn.DataParallel(net)

    # loss init
    det_criterion = MultiBoxLoss(
        cfg['det_loss']['num_classes'],
        cfg['det_loss']['overlap_thresh'],
        cfg['det_loss']['prior_for_matching'],
        cfg['det_loss']['bkg_label'],
        cfg['det_loss']['neg_mining'],
        cfg['det_loss']['neg_pos'],
        cfg['det_loss']['neg_overlap'],
        cfg['det_loss']['encode_target'],
        cfg['det_loss']['use_gpu']
    )
    criterion = nn.CrossEntropyLoss()

    # optimizer init.
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

    # load checkpoint if given
    base_epoch = 0
    if args.ckpt:
        net, optimzer, base_epoch = load_checkpoint(args.ckpt, net, optimizer, local_rank)

    # get training data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    cfg['sbi_key'] = args.sbi
    train_dataset = DeepfakeDataset('train', cfg)
    sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'] // 4,
                              num_workers=4,
                              collate_fn=my_collate,
                              pin_memory=True,
                              sampler=sampler
                              )
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    local_rank = int(os.environ['LOCAL_RANK'])
    net = torch.nn.parallel.DistributedDataParallel(net, 
                                                    device_ids=[local_rank],
                                                    find_unused_parameters=True)
    # start trining.
    net.train()
    for epoch in range(base_epoch, cfg['train']['epoch_num']):
        train_loader.sampler.set_epoch(epoch)
        mini_batch = 0
        for index, (batch_data, batch_labels) in enumerate(train_loader):

            lr = update_learning_rate(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            labels, location_labels, confidence_labels = batch_labels
            labels = labels.long().to(device)
            location_labels = location_labels.to(device)
            confidence_labels = confidence_labels.long().to(device)

            optimizer.zero_grad()
            locations, confidence, outputs = net(batch_data)
            loss_end_cls = criterion(outputs, labels)
            loss_l, loss_c = det_criterion(
                (locations, confidence),
                confidence_labels, location_labels
            )
            acc = sum(outputs.max(-1).indices == labels).item() / labels.shape[0]
            det_loss = 0.1 * (loss_l + loss_c)
            loss = det_loss + loss_end_cls
            loss.backward()

            torch.nn.utils.clip_grad_value_(net.parameters(), 2)
            optimizer.step()

            outputs = [
                "e:{},iter: {}".format(epoch, index),
                "acc: {:.2f}".format(acc),
                "loss: {:.8f} ".format(loss.item()),
                "lr:{:.4g}".format(lr),
            ]
            print(" ".join(outputs))
            mini_batch += 1

            if mini_batch == 128:
                break
        if torch.distributed.get_rank() == 0 and epoch % 10 == 0:
            save_checkpoint(net, optimizer,
                            cfg['model']['save_path'],
                            epoch)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    train()

# vim: ts=4 sw=4 sts=4 expandtab
