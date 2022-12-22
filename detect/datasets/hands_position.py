import os
from unicodedata import decimal
import numpy as np
import cv2
import argparse
from pathlib import Path
from PIL import Image
import torch

import sys
sys.path.append("..")
import os

# conda activate yolov5
# python detect/datasets/hands_position.py --datasets_dir E:/datasets/my_hand_pose_dataset/train
# python detect/datasets/hands_position.py --datasets_file E:/datasets/my_hand_pose_dataset/train/video/1_finger_1_1.mp4

parser = argparse.ArgumentParser()
parser.add_argument("--frame_interval", type=int, default=5, help="get frame landmark every frame_interval value")
parser.add_argument("--datasets_dir", type=str, default='E:/datasets/my_hand_pose_dataset/train', help="give the datasets dir")
parser.add_argument("--datasets_file", type=str, help="give the datasets video file")

opt = parser.parse_args()
print(opt)

model = torch.hub.load('ultralytics/yolov5', 'custom', 
    path='f:/repositories/detect_object/yolov5/runs/train/exp10/weights/best.pt')  # or yolov5m, yolov5l, yolov5x, custom

def hands_position():

    total = 0
    if opt.datasets_file is not None:
        dir = Path(opt.datasets_file).parent.absolute()
        result_image_dir = os.path.join(dir, 'images')
        Path(result_image_dir).mkdir(parents=True, exist_ok=True)
        result_label_dir = os.path.join(dir, 'labels')
        Path(result_label_dir).mkdir(parents=True, exist_ok=True)

        total = hands_position_file(opt.datasets_file, result_image_dir, result_label_dir)
    elif os.path.isdir(opt.datasets_dir):
        result_image_dir = os.path.join(opt.datasets_dir, 'images')
        Path(result_image_dir).mkdir(parents=True, exist_ok=True)
        result_label_dir = os.path.join(opt.datasets_dir, 'labels')
        Path(result_label_dir).mkdir(parents=True, exist_ok=True)

        video_file_dir = os.path.join(opt.datasets_dir, 'video')
        for video_file in os.listdir(video_file_dir):
            filename = os.fsdecode(video_file)
            if not filename.lower().endswith(".mp4"): continue
            total += hands_position_file(os.path.join(video_file_dir, video_file), result_image_dir, result_label_dir)

    print('Total image count: {}'.format(total))

# no effect no use diff
def hands_position_file(video_file, result_image_dir, result_label_dir):
    video_filename = Path(video_file).stem
    target = video_filename.split('_')[0]
    assert target.isdigit(), 'target is not a number.'
    target = int(target) - 1
    
    cap = cv2.VideoCapture(video_file)
    frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index, result_index, cut_off_begin, cut_off_end = -1, 0, 100, 100
    while cap.isOpened():

        success, image = cap.read()
        frame_index += 1

        if not success: continue
        if frame_index > frame_len - cut_off_end: break

        # skip frame less then start frame and within the frame interval 
        if frame_index < cut_off_begin or frame_index % opt.frame_interval != 0: continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = image.shape[1], image.shape[0]

        # Inference
        results = model(image) 

        if results.pred[0].shape[0] == 1:

            xyxy = results.tolist()[0].xyxy[0].cpu().numpy().copy().squeeze()[:4]
            xywh = results.tolist()[0].xywh[0].cpu().numpy().copy().squeeze()[:4]

            # get max in wh and compute spread
            ex_scale, offset_top_scale = 0.05, 0.1

            xyxy_new = xyxy.copy()
            xyxy_new[:2] = xyxy_new[:2] - xywh[2:] * ex_scale
            xyxy_new[2:] = xyxy_new[2:] + xywh[2:] * ex_scale

            # set top offset
            xyxy_new[1] = xyxy_new[1] - xywh[2] * offset_top_scale
            xyxy_new[3] = xyxy_new[3] - xywh[3] * offset_top_scale

            xywh_ex = np.array([xyxy_new[0] / w, xyxy_new[1] / h, (xyxy_new[2] - xyxy_new[0]) / w, (xyxy_new[3] - xyxy_new[1]) / h], dtype=np.float32)
            
            # set to center
            xywh_ex[0] = xywh_ex[0] + xywh_ex[2] / 2
            xywh_ex[1] = xywh_ex[1] + xywh_ex[3] / 2

            xywh_ex = np.where(xywh_ex < 0, 0, xywh_ex)
            xywh_ex = np.where(xywh_ex > 1, 1, xywh_ex)
            np.round(xywh_ex, decimals=6)

            filename = os.path.join(result_label_dir, video_filename + '_{}.txt'.format(result_index))
            open(filename, "w").write(str(target) + ' ' + ' '.join(str(round(s, 6)) for s in xywh_ex.tolist()))

            image = Image.fromarray(image)
            image = image.resize((image.width // 3, image.height // 3))
            image.save(os.path.join(result_image_dir, video_filename + '_{}.jpg'.format(result_index)))

            result_index += 1
            print('Video file: {}, get hand image count: {}.'.format(video_file, result_index), end='\r')
    
    print()
    return result_index

if __name__ == '__main__':
    hands_position()      