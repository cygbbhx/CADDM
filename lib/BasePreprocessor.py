#!/usr/bin/env python3
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import dlib
import json
from imutils import face_utils
from abc import ABCMeta, abstractmethod
import pandas as pd 

class BasePreprocessor(metaclass=ABCMeta):
    def __init__(self, data_root, save_root, num_frames, predictor_path):
        self.data_root = data_root
        self.save_root = save_root
        self.num_frames = num_frames
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(predictor_path)
        self.img_meta_dict = {}

    @abstractmethod
    def parse_labels(self, video_path):
        pass

    @abstractmethod
    def get_video_paths(self):
        pass

    # source (original version) of fake video 
    # IF not implemented, returns the fake video
    def parse_source_save_path(self, save_path):
        return save_path

    def preprocess_video(self, video_path, save_path):
        video_dict = dict()
        label = self.parse_labels(video_path)
        source_save_path = self.parse_source_save_path(save_path)
        os.makedirs(save_path, exist_ok=True)

        cap_video = cv2.VideoCapture(video_path)
        frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, frame_count_video - 1, self.num_frames, endpoint=True, dtype=np.int)
        
        for cnt_frame in range(frame_count_video):
            ret, frame = cap_video.read()
            if not ret:
                tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(video_path)))
                continue
            if cnt_frame not in frame_idxs:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(frame, 1)

            if len(faces) == 0:
                tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(video_path)))
                continue
            landmarks = []
            size_list = []

            for face_idx in range(len(faces)):
                landmark = self.face_predictor(frame, faces[face_idx])
                landmark = face_utils.shape_to_np(landmark)
                x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
                x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
                face_s = (x1 - x0) * (y1 - y0)
                size_list.append(face_s)
                landmarks.append(landmark)
            landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
            landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]
            
            video_dict['landmark'] = landmarks.tolist()
            video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"
            video_dict['label'] = label
            self.img_meta_dict[f"{save_path}/frame_{cnt_frame}"] = video_dict
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image_path = f"{save_path}/frame_{cnt_frame}.png"
            cv2.imwrite(image_path, frame)
        cap_video.release()
        return

    def preprocess_dataset(self):
        videos_path_list = self.get_video_paths()
        print("{} : videos are exist in {}".format(len(videos_path_list), self.data_root))
        n_videos = len(videos_path_list)

        for i in tqdm(range(n_videos)):
            save_path_per_video = videos_path_list[i].replace(
                self.data_root, self.save_root
            ).replace('.mp4', '')
            self.preprocess_video(videos_path_list[i], save_path_per_video)
        with open(f"{self.save_root}/ldm.json", 'w') as f:
            json.dump(self.img_meta_dict, f)


class FFProcessor(BasePreprocessor):
    def parse_labels(self, video_path):
        if "original" in video_path:
            return 0
        else:
            return 1
        
    def parse_source_save_path(self, save_path):
        if "original" in save_path:
            return save_path
        else:
            img_meta = save_path.split('/')
            source_target_index = img_meta[-1]
            manipulation_name = img_meta[-4]
            original_name = "youtube"
            source_index = source_target_index.split('_')[0]
            source_save_path = save_path.replace(
                "manipulated_sequences", "original_sequences"
            ).replace(
                manipulation_name, original_name
            ).replace(
                source_target_index, source_index
            )
            return source_save_path
        
    def get_video_paths(self):
        datasets = ['Original', 'FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']
        compressions = ['raw']
        videos_path_list = []

        for dataset in datasets:
            for comp in compressions:
                if dataset == 'Original':
                    dataset_path = f'{self.data_root}/original_sequences/youtube/{comp}/videos/'
                elif dataset in ['FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
                    dataset_path = f'{self.data_root}/manipulated_sequences/{dataset}/{comp}/videos/'
                else:
                    raise NotImplementedError
                videos_path_list.extend(sorted(glob(dataset_path + '*.mp4')))
        return videos_path_list

class CelebProcessor(BasePreprocessor):
    def __init__(self, data_root, save_root, num_frames, predictor_path):
        super().__init__(data_root, save_root, num_frames, predictor_path)
        label_path = os.path.join(self.data_root, 'List_of_testing_videos.txt')
        self.labels = {}
        with open(label_path, 'r') as file:
            for line in file:
                line = line.strip()
                label, path = line.split(maxsplit=1)
                # Celeb labels are reversed (0 Fake 1 Real -> 0 Real 1 Fake)
                self.labels[path.replace('.mp4', '')] = int(label) ^ 1 
        
    def parse_labels(self, video_path):
        video_key = os.path.relpath(video_path, self.data_root).replace('.mp4', '')
        return self.labels[video_key]

    def parse_source_save_path(self, save_path):
        if 'Celeb-synthesis' in save_path:
            # Celeb-synthesis/.../id0_id1_0000 -> Celeb-real/.../id0_0000
            return save_path.replace('synthesis', 'real').replace(save_path.split('_')[1], "")
        else:
            return save_path
    
    def get_video_paths(self):
        all_video_paths = sorted(glob(f'{self.data_root}/*/*.mp4'))
        test_video_paths = [video_path for video_path in all_video_paths 
                            if os.path.relpath(video_path, self.data_root).replace('.mp4', '') in self.labels.keys()]
        return test_video_paths
    
class DFDCProcessor(BasePreprocessor):
    def __init__(self, data_root, save_root, num_frames, predictor_path):
        super().__init__(data_root, save_root, num_frames, predictor_path)
        csv_path = os.path.join(self.data_root, 'labels.csv')
        labels_df = pd.read_csv(csv_path)
        self.labels = {filename.replace('.mp4', ''): label for filename, label in 
                       zip(labels_df["filename"], labels_df["label"])}

    # No source label for DFDC test set, omit source_save_path
    def parse_labels(self, video_path):
        video_key = os.path.basename(video_path).replace('.mp4', '')
        return self.labels[video_key]
    
    def get_video_paths(self):
        return sorted(glob(f'{self.data_root}/*/*.mp4'))

def main():    
    DATA_ROOT = "/workspace/dataset/FaceForensics++/"
    SAVE_ROOT = "./train_images/"
    NUM_FRAMES = 32
    PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"

    os.makedirs(SAVE_ROOT, exist_ok=True)

    dataset_processor = FFProcessor(DATA_ROOT, SAVE_ROOT, NUM_FRAMES, PREDICTOR_PATH)
    dataset_processor.preprocess_dataset()

if __name__ == '__main__':
    main()
