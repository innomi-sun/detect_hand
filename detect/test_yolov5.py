import sys
import cv2
import numpy as np
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont

# python detect/test_yolov5.py --target "E:/datasets/my_hand_pose_dataset/test/video/test_020220215_171325.mp4"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default='detect/checkpoint/ckp_2022_03_14_1407_resnet18_86.7.pth', help="checkpoint path")
parser.add_argument("--target", type=str, default='0', help="video path of test target, use default camera when set to '0' or nothing. ")
parser.add_argument("--draw_landmarks", action='store_true', help="draw landmarks")

parser.add_argument("--num_class", type=int, default=21, help="number of epochs of training")
parser.add_argument("--input_size", type=int, default=21, help="number of epochs of training")

opt = parser.parse_args()
print(opt)
    
def test():

    model = torch.hub.load('ultralytics/yolov5', 'custom', 
        path='f:/repositories/detect_object/yolov5/runs/train/exp22/weights/best.pt')

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 1)

        # Inference
        results = model(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pred[0].shape[0] == 1:

            xyxy = results.tolist()[0].xyxy[0].cpu().numpy().copy().squeeze()[:4]
            # xywh = results.tolist()[0].xywh[0].cpu().numpy().copy().squeeze()[:4]

            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.rectangle(xyxy, outline='#ff0000')

            font = ImageFont.truetype(r'C:/Windows/Fonts/arial.ttf', 40)
            cls = int(results.tolist()[0].xyxy[0].cpu().numpy().copy().squeeze()[5:][0])
            draw.text((20, 20), results.tolist()[0].names[cls], fill='#00ff00', font=font)
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Hands', np.array(image))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

if __name__ == '__main__':
    test()    