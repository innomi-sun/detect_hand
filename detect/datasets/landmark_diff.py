import os
import numpy as np
import cv2
import argparse
from pathlib import Path

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

"""
from mediapipe landmarks 21, xyz 3 to 28(landmarks) * 3(x,y,z) by add 7 * 3 all 0 value
get clip 5 frames(frames_count) as image data

21 hand landmarks and each landmark is composed of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
z represents the landmark depth with the depth at the wrist being the origin, 
and the smaller the value the closer the landmark is to the camera. 
The magnitude of z uses roughly the same scale as x.

x -> [0.0, 1.0] -> [0, 255]
y -> [0.0, 1.0] -> [0, 255]
z -> [(]close to camera, far away from camera]-> [-1.0, 0.0] -> [0, 255]

"""

# python detect/datasets/landmark_diff.py --datasets_dir E:/datasets/my_hand_pose_dataset/train
# python detect/datasets/landmark_diff.py --datasets_file E:/datasets/my_hand_pose_dataset/train/video/9_back_hand_20220222_153925.mp4

parser = argparse.ArgumentParser()
parser.add_argument("--frame_interval", type=int, default=5, help="get frame landmark every frame_interval value")
parser.add_argument("--datasets_dir", type=str, default='E:/datasets/my_hand_pose_dataset/val', help="give the datasets dir")
parser.add_argument("--datasets_file", type=str, help="give the datasets video file")

opt = parser.parse_args()
print(opt)

def landmark_diff():

    total = 0
    if opt.datasets_file is not None:
        anno_file_dir = os.path.dirname(os.path.realpath(opt.datasets_file))
        total = landmark_diff_file(opt.datasets_file, anno_file_dir)
    else:
        video_file_dir = os.path.join(opt.datasets_dir, 'video')
        anno_file_dir = os.path.join(opt.datasets_dir, 'annotations')
        Path(anno_file_dir).mkdir(parents=True, exist_ok=True)

        for video_file in os.listdir(video_file_dir):
            filename = os.fsdecode(video_file)
            if not filename.lower().endswith(".mp4"): 
                continue
            total += landmark_diff_file(os.path.join(video_file_dir, video_file), anno_file_dir)

    print('Total landmark frames: {}'.format(total))

# no effect no use diff
def landmark_file(video_file, anno_file_dir):

    total, cut_off_begin, cut_off_end = 0, 100, 100
    filename, data = os.fsdecode(video_file), None
    assert filename.lower().endswith(".mp4"), 'can not found a mp4 file.'
    
    file_name = os.path.splitext(os.path.basename(video_file))[0]
    target = file_name.split('_')[0]
    assert target.isdigit(), 'target is not a number.'
    target = int(target)

    cap = cv2.VideoCapture(video_file)
    frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    frame_index, landmark_diff_len, landmark_count = 0, 20, 21
    # target_total_frame = (frame_range[1] - frame_range[0] ) // opt.frame_interval
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        frame_index += 1
        if frame_index > frame_len - cut_off_end:
            break
        
        # skip frame less then start frame and within the frame interval 
        if frame_index < cut_off_begin or frame_index % opt.frame_interval != 0:
            continue

        # To improve performance, optionally mark the image as not writeable to
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            # only get first hand landmarks diff
            #hand_landmarks = results.multi_hand_landmarks[0]

            hand_landmarks = results.multi_hand_world_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                item = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                landmarks.append(item)
            
            # get landmarks, add landmark_8 - landmark_4, landmark_12 - landmark_8, landmark_16 - landmark_12, landmark_20 - landmark_16
            landmarks_result = []
            for key in ('x', 'y', 'z'):
                for i in range(landmark_count):
                    landmarks_result.append(landmarks[i][key])
                landmarks_result.append(landmarks[8][key] - landmarks[4][key])
                landmarks_result.append(landmarks[12][key] - landmarks[8][key])
                landmarks_result.append(landmarks[16][key] - landmarks[12][key])
                landmarks_result.append(landmarks[20][key] - landmarks[16][key])
        else:
            continue

        result = np.array(landmarks_result)
        result = result.reshape(25 * 3)

        # normalize landmark difference data to [0, 1] on x, y, z
        result[:25] = cv2.normalize(result[:25], result[:25], norm_type=cv2.NORM_MINMAX)
        result[25:50] = cv2.normalize(result[25:50], result[25:50], norm_type=cv2.NORM_MINMAX)
        result[50:75] = cv2.normalize(result[50:75], result[50:75], norm_type=cv2.NORM_MINMAX)

        result = np.reshape(result, (1, 5, 5, 3))
        result = (result * 255).astype(np.uint8)
            
        if type(data).__module__ != np.__name__ and data == None:
            data = result.copy()
        else:
            data = np.concatenate((data, result), axis=0)

        print('Video file: {}, get landmark frames count: {}.'.format(video_file, data.shape[0]), end='\r')
    
    #print('Video file: {}, processing images: {}/{}. Get hand landmark frame total: {}'.format(video_file, data.shape[0], target_total_frame, data.shape[0]))
    print()
    if type(data).__module__ == np.__name__:
        total = data.shape[0]
        data.tofile(os.path.join(anno_file_dir, file_name + '.txt'), sep=',', format='%s')
    cap.release()

    return total

# get clip landmark annotations from video
def landmark_diff_file(video_file, anno_file_dir):

    total, cut_off_begin, cut_off_end = 0, 100, 100
    filename, data = os.fsdecode(video_file), None
    assert filename.lower().endswith(".mp4"), 'can not found a mp4 file.'
    
    file_name = os.path.splitext(os.path.basename(video_file))[0]
    target = file_name.split('_')[0]
    assert target.isdigit(), 'target is not a number.'
    target = int(target)

    cap = cv2.VideoCapture(video_file)
    frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    frame_index, landmark_diff_len, landmark_count = 0, 21, 21
    # target_total_frame = (frame_range[1] - frame_range[0] ) // opt.frame_interval
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        frame_index += 1
        if frame_index > frame_len - cut_off_end:
            break
        
        # skip frame less then start frame and within the frame interval 
        if frame_index < cut_off_begin or frame_index % opt.frame_interval != 0:
            continue

        # To improve performance, optionally mark the image as not writeable to
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            # only get first hand landmarks diff
            #hand_landmarks = results.multi_hand_landmarks[0]

            hand_landmarks = results.multi_hand_world_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                item = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                landmarks.append(item)
            
            # get difference of landmarks, total is 21!/19!*2! in xyz, 
            # except 0 and others landmarks's difference to make 20 * 20 result
            # result is 400(landmarks diff) * 3 (xyz)
            landmarks_all, landmarks_diff_minus, landmarks_diff_plus = [], [], []
            for key in ('x', 'y', 'z'):
                for i in range(landmark_count):
                    landmarks_all.append(landmarks[i][key])
                    for j in range(i + 1, landmark_count):
                        landmarks_diff_minus.append(landmarks[i][key] - landmarks[j][key])
                        landmarks_diff_plus.append(landmarks[i][key] + landmarks[j][key])
        else:
            continue

        # normalize landmark difference data to [0, 1] on x, y, z
        len = 210
        result = np.array(landmarks_all)
        result_minus = np.array(landmarks_diff_minus)
        result_plus = np.array(landmarks_diff_plus)
        for i in range(3):
            result[i * landmark_count: (i + 1) * landmark_count] = cv2.normalize(result[i * landmark_count: (i + 1) * landmark_count], result[i * landmark_count: (i + 1) * landmark_count], norm_type=cv2.NORM_MINMAX)
            result_minus[i * len: (i + 1) * len] = cv2.normalize(result_minus[i * len: (i + 1) * len], result_minus[i * len: (i + 1) * len], norm_type=cv2.NORM_MINMAX)
            result_plus[i * len: (i + 1) * len] = cv2.normalize(result_plus[i * len: (i + 1) * len], result_plus[i * len: (i + 1) * len], norm_type=cv2.NORM_MINMAX)

        result = np.concatenate((result, result_minus, result_plus), axis=0)
        result = np.reshape(result, (1, landmark_diff_len, -1, 3))
        result = (result * 255).astype(np.uint8)
            
        if type(data).__module__ != np.__name__ and data == None:
            data = result.copy()
        else:
            data = np.concatenate((data, result), axis=0)

        print('Video file: {}, get landmark frames count: {}.'.format(video_file, data.shape[0]), end='\r')
    
    #print('Video file: {}, processing images: {}/{}. Get hand landmark frame total: {}'.format(video_file, data.shape[0], target_total_frame, data.shape[0]))
    print()
    if type(data).__module__ == np.__name__:
        total = data.shape[0]
        data.tofile(os.path.join(anno_file_dir, file_name + '.txt'), sep=',', format='%s')
    cap.release()

    return total

if __name__ == '__main__':
    landmark_diff()      