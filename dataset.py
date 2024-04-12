#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import re 
from tqdm import tqdm
from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input
import albumentations as alb
from sbi.funcs import IoUfrom2bboxes,crop_face,RandomDownScale
from sbi import blend as B


class DeepfakeDataset(Dataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        mode: train or test.
        config: hypter parameters for processing images.
    """

    def __init__(self, mode: str, config: dict):
        super().__init__()

        self.config = config
        self.mode = mode
        self.root = self.config['dataset']['img_path']
        self.landmark_path = self.config['dataset']['ld_path']
        self.rng = np.random
        assert mode in ['train', 'test']
        self.do_train = True if mode == 'train' else False
        self.info_meta_dict = self.load_landmark_json(self.landmark_path)

        self.sbi_key = self.config['sbi_key']
        self.source_transforms = self.get_source_transforms()

        self.class_dict = self.collect_class()
        self.samples = self.collect_samples()

    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict

    def collect_samples(self) -> List:
        samples = []
        directory = os.path.expanduser(self.root)
        for key in sorted(self.class_dict.keys()):
            d = os.path.join(directory, key)
            if not os.path.isdir(d):
                continue
            for r, _, filename in sorted(os.walk(d, followlinks=True)):
                for name in sorted(filename):
                    path = os.path.join(r, name)
                    info_key = path[:-4]
                    video_name = '/'.join(path.split('/')[:-1])
                    if info_key not in self.info_meta_dict.keys():
                        continue
                    info_meta = self.info_meta_dict[info_key]
                    landmark = info_meta['landmark']
                    class_label = int(info_meta['label'])
                    source_path = info_meta['source_path'] + path[-4:]
                    samples.append(
                        (path, {'labels': class_label, 'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name})
                    )

        return samples

    def collect_class(self) -> Dict:
        if 'dfdc' in self.root:
            return {'': 0}
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort(reverse=True)
        
        if self.sbi_key != 0:
            classes = [dataset for dataset in classes \
                       if all(m not in dataset for m in ['Face2Face', 'FaceSwap', 'NeuralTextures'])]
            # ensure we have only original_sequences(REAL) and DeepFake(FAKE). 
            # DeepFake is just dummy fake data, will be replaced by SBI images from real
            assert len(classes) == 2, f"only original and manipulated-DeepFakes allowd in SBI, found {classes}"

        return {classes[i]: np.int32(i) for i in range(len(classes))}

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        source_path = label_meta['source_path']
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if not os.path.exists(source_path):
            directory = os.path.dirname(source_path)
            filename = os.path.basename(source_path)

            nearest_filename = self.find_nearest_filename(directory, filename)
            source_path = os.path.join(directory, nearest_filename)

        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if self.mode == "train":
            if self.sbi_key == 1 and label == 1: # Swap After SBI
                _, img, _ = self.self_blending(img.copy(),ld.copy())

            img, label_dict = prepare_train_input(
                img, source_img, ld, label, self.config, self.do_train
            )
            if isinstance(label_dict, str):
                return None, label_dict

            location_label = torch.Tensor(label_dict['location_label'])
            confidence_label = torch.Tensor(label_dict['confidence_label'])
            if self.sbi_key == 2 and label == 1: # SBI After Swap
                _, img, _ = self.self_blending(img.copy(),ld.copy())            

            img = torch.Tensor(img.transpose(2, 0, 1))            
            return img, (label, location_label, confidence_label)

        elif self.mode == 'test':
            img, label_dict = prepare_test_input(
                [img], ld, label, self.config
            )
            img = torch.Tensor(img[0].transpose(2, 0, 1))
            video_name = label_meta['video_name']
            return img, (label, video_name)

        else:
            raise ValueError("Unsupported mode of dataset!")

    def __len__(self):
        return len(self.samples)

    def find_nearest_filename(self, directory, filename):
        number = re.search(r'\d+', filename).group()
        number = int(number)

        files = os.listdir(directory)
        files_with_numbers = [f for f in files if re.search(r'\d+', f)]
        numbers = sorted([int(re.search(r'\d+', f).group()) for f in files_with_numbers])
        nearest_number = min(numbers, key=lambda x: abs(x - number))
        nearest_filename = [f for f in files_with_numbers if str(nearest_number) in f][0]

        return nearest_filename
    
    ### Imported from SBI ###
    def self_blending(self,img,landmark):
        H,W=len(img),len(img[0])
        if np.random.rand()<0.25:
            landmark=landmark[:68]
        mask=np.zeros_like(img[:,:,0])
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        if np.random.rand()<0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source,mask)

        img_blended,mask=B.dynamic_blend(source,img,mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img,img_blended,mask
    
    def get_source_transforms(self):
        return alb.Compose([
                alb.Compose([
                        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                        alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                    ],p=1),

                alb.OneOf([
                    RandomDownScale(p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ],p=1),
                
            ], p=1.)
    

    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']

        mask=transformed['mask']
        transformed=g(image=img,mask=mask)
        mask=transformed['mask']
        return img,mask


if __name__ == "__main__":
    from lib.util import load_config
    config = load_config('./configs/caddm_train.cfg')
    d = DeepfakeDataset(mode="train", config=config)
    for index in tqdm(range(len(d))):
        res = d[index]
# vim: ts=4 sw=4 sts=4 expandtab
