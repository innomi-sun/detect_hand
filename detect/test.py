import sys
import cv2
import numpy as np
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from detector import Detector

# python detect/test.py --draw_landmarks --target "E:/datasets/my_hand_pose_dataset/test/video/test_020220215_171325.mp4"
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default='detect/checkpoint/ckp_2022_03_14_1407_resnet18_86.7.pth', help="checkpoint path")
parser.add_argument("--target", type=str, default='0', help="video path of test target, use default camera when set to '0' or nothing. ")
parser.add_argument("--draw_landmarks", action='store_true', help="draw landmarks")

parser.add_argument("--num_class", type=int, default=21, help="number of epochs of training")
parser.add_argument("--input_size", type=int, default=21, help="number of epochs of training")

opt = parser.parse_args()
print(opt)

def test():
    
    detector = Detector(opt.checkpoint, min_confidence=0.5)

    pos_frames, label_frames = [], []
    frame_index, skip_frames, count_to_skip, landmark_count, detect_max_frame = 0, 0, 30, 21, 30

    font, fontScale, fontColor, thickness, lineType = cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, 2

    if opt.target == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(opt.target)

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8)
        
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        skip_frames -= 1
        frame_index += 1
        landmarks, landmarks_location = [], []
        landmarks_all, landmarks_diff_minus, landmarks_diff_plus = [], [], []
        if results.multi_hand_landmarks:
            # only get first hand landmarks diff
            hand_landmarks = results.multi_hand_landmarks[0]
            for landmark in hand_landmarks.landmark:
                landmarks_location.append(landmark.x)
                landmarks_location.append(landmark.y)
                landmarks_location.append(landmark.z)

            hand_world_landmarks = results.multi_hand_world_landmarks[0]
            for landmark in hand_world_landmarks.landmark:
                item = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                landmarks.append(item)
            
            # get difference of landmarks, count is 21!/19!*2! in xyz, 
            # except 0 and others landmarks's difference to make 20 * 20 result
            # result is 400(landmarks diff) * 3 (xyz)

            for key in ('x', 'y', 'z'):
                for i in range(landmark_count):
                    landmarks_all.append(landmarks[i][key])
                    for j in range(i + 1, landmark_count):
                        landmarks_diff_minus.append(landmarks[i][key] - landmarks[j][key])
                        landmarks_diff_plus.append(landmarks[i][key] + landmarks[j][key])
    
            if opt.draw_landmarks:
                mp_drawing.draw_landmarks(    
                image,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


        if len(landmarks_all) > 0:
            # normalize landmark difference data to [0, 1] on x, y, z
            length = 210
            result = np.array(landmarks_all)
            result_minus = np.array(landmarks_diff_minus)
            result_plus = np.array(landmarks_diff_plus)
            for i in range(3):
                result[i * landmark_count: (i + 1) * landmark_count] = cv2.normalize(result[i * landmark_count: (i + 1) * landmark_count], result[i * landmark_count: (i + 1) * landmark_count], norm_type=cv2.NORM_MINMAX)
                result_minus[i * length: (i + 1) * length] = cv2.normalize(result_minus[i * length: (i + 1) * length], result_minus[i * length: (i + 1) * length], norm_type=cv2.NORM_MINMAX)
                result_plus[i * length: (i + 1) * length] = cv2.normalize(result_plus[i * length: (i + 1) * length], result_plus[i * length: (i + 1) * length], norm_type=cv2.NORM_MINMAX)

            result = np.concatenate((result, result_minus, result_plus), axis=0)
            result = np.reshape(result, (21, -1, 3))
            result = (result * 255).astype(np.uint8)
                
            label_pred, confidence_pred, prob, ret = detector.predict_landmarks(result)
            # print(label_pred, confidence_pred, prob, ret)
            if label_pred != -1: # and skip_frames <= 0:
                # print(label_pred, confidence_pred.round(decimals=3), prob.round(decimals=2), ret.round(decimals=2), sep='\t')
                cv2.putText(image, str(label_pred + 1), (10, 60), font, fontScale, fontColor, thickness, lineType)
            
                label_frames.append(label_pred)
                pos_frames.append(landmarks_location.copy())

                # get the detect_max_frame data pass to detector
                if len(label_frames) >= detect_max_frame:
                    labels = np.array(label_frames[len(label_frames) - detect_max_frame: len(label_frames)].copy())
                    pos = np.array(pos_frames[len(pos_frames) - detect_max_frame: len(pos_frames)].copy())
                    #pos = np.transpose(pos.reshape(detect_max_frame, -1, 3), (0, 2, 1))
                    pos = pos.reshape(detect_max_frame, landmark_count, 3)

                    # get subtract of all frames and sum them as offset at every dimension(x, y, z)
                    pos_diff = pos[1:] - pos[:detect_max_frame - 1]
                    offset_xyz = pos_diff.sum(axis=0).sum(axis=0)

                    # action = detector.detect_action(labels)
                    # if action:
                    #     skip_frames = count_to_skip
                    #     cv2.putText(image, str(action), (50, 60), font, fontScale, fontColor, thickness, lineType)

                    print(label_pred, offset_xyz)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

if __name__ == '__main__':
    test()    