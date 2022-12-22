import os
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
# python detect/datasets/hands_cut.py --datasets_dir E:/datasets/my_hand_pose_dataset/train
# python detect/datasets/hands_cut.py --datasets_file E:/datasets/my_hand_pose_dataset/train/video/1_finger_1_1.mp4

parser = argparse.ArgumentParser()
parser.add_argument("--frame_interval", type=int, default=2, help="get frame landmark every frame_interval value")
parser.add_argument("--datasets_dir", type=str, default='E:/datasets/my_hand_pose_dataset/train/video', help="give the datasets dir")
parser.add_argument("--datasets_file", type=str, help="give the datasets video file")

opt = parser.parse_args()
print(opt)

def hands_cut():

    total = 0
    if opt.datasets_file is not None:

        result_image_dir = os.path.splitext(opt.datasets_file)[0]
        Path(result_image_dir).mkdir(parents=True, exist_ok=True)
        total = hands_cut_file(opt.datasets_file, result_image_dir)

    else:

        video_file_dir = os.path.join(opt.datasets_dir, 'video')
        result_image_dir = os.path.join(opt.datasets_dir, 'images')

        for video_file in os.listdir(video_file_dir):

            filename = os.fsdecode(video_file)
            result_image_dir = os.path.join(result_image_dir, os.path.splitext(filename)[0])
            Path(result_image_dir).mkdir(parents=True, exist_ok=True)

            if not filename.lower().endswith(".mp4"): continue
            total += hands_cut_file(os.path.join(video_file_dir, video_file), result_image_dir)

    print('Total image count: {}'.format(total))

# no effect no use diff
def hands_cut_file(video_file, result_image_dir):

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='f:/repositories/detect_object/yolov5/runs/train/exp12/weights/best.pt')  # or yolov5m, yolov5l, yolov5x, custom
    
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

        # Inference
        results = model(image) 

        if results.pred[0].shape[0] == 1:

            xyxy = results.tolist()[0].xyxy[0].cpu().numpy().copy().squeeze()[:4]
            xywh = results.tolist()[0].xywh[0].cpu().numpy().copy().squeeze()[:4]

            # get max in wh and compute spread
            ex_scale, offset_top_scale = 0.2, 0.2
            ex = xywh[2:].max() * ex_scale

            # set top offset
            offset_top = xywh[2:].max() * offset_top_scale
            xyxy[1] = xyxy[1] - offset_top

            # use max in wh to expend to square
            wh_max = xywh[2:].max() + ex * 2
            xyxy_ex = np.array([xyxy[0] - ex, xyxy[1] - ex, xyxy[0] - ex + wh_max, xyxy[1] - ex + wh_max], dtype=np.float32)

            image = Image.fromarray(image)
            # draw = ImageDraw.Draw(image)
            # draw.rectangle(xyxy, outline='#ff0000')

            img_crop = image.crop(xyxy_ex)
            img_crop.save(os.path.join(result_image_dir, '{}.jpg'.format(result_index)))
            result_index += 1
            print('Video file: {}, get hand image count: {}.'.format(video_file, result_index), end='\r')
    
    print()
    return result_index

if __name__ == '__main__':
    hands_cut()      