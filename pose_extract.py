import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
import json
import gc
import sys
from config import DATA_DIR, MAX_FRAMES, CLASS_MAPPING_FILE


if sys.platform.startswith('win'):
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)


def extract_hand_and_arm_landmarks(video_path,max_frames):
    '''
    提取手部和手臂关键点
    '''
    mp_hands = mp.solutions.hands
    mp_arm = mp.solutions.pose
    
    landmarks_sequence = []
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    with mp.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as hands , \
    mp_arm.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as pose:
        while cap.isOpened() and frame_count < max_frames:
            
            ret,frame = cap.read()
            if not ret:
                break
            
            #图像转换为RGB
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            hand_result = hands.process(image)
            pose_result = pose.process(image)
            
            # 提取关键点
            
            feature = []
            
            if hand_result.multi_hand_landmarks:
                left_hand_detected = False
                right_hand_detected = False
                
                for hand_landmarks,handedness in zip(hand_result.multi_hand_landmarks,hand_result.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    if hand_label == "Left" and not left_hand_detected:
                        for landmark in hand_landmarks.landmark:
                            feature.extend([landmark.x,landmark.y])
                            left_hand_detected = True
                    elif hand_label == "Right" and not right_hand_detected:
                        for landmark in hand_landmarks.landmark:
                            feature.extend([landmark.x,landmark.y])
                            right_hand_detected = True
                            
                if not left_hand_detected:
                    feature.extend([0,0] * 21)
                
                if not right_hand_detected:
                    feature.extend([0,0] * 21)
                    
            else:
                feature.extend([0,0]*42)
            
            if pose_result.pose_landmarks:
                arm_indices = [11,13,15,12,14,16]
                # 左肩 (11), 左肘 (13), 左腕 (15)
                # 右肩 (12), 右肘 (14), 右腕 (16)
                
                for idx in arm_indices:
                    landmark = pose_result.pose_landmarks.landmark[idx]
                    feature.extend([landmark.x,landmark.y])
            else:
                feature.extend([0,0] * 6)
            
            landmarks_sequence.append(feature)
            frame_count += 1
            
    cap.release()
    return np.array(landmarks_sequence)



def extract_all_features(datadir=DATA_DIR,max_frames=MAX_FRAMES):
    '''提取所有视频的特征'''
    
    with open("./class_mapping.json",'r') as f:
        class_mapping = json.load(f)
    class_to_idx = class_mapping['class_to_idx']
    
    X = [] # 特征
    y = [] # 标签
    
    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_idx = class_to_idx[class_name]
        print(f"处理类别: {class_name} (索引: {class_idx})")

        video_files =[f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_file in tqdm(video_files):
            video_path = os.path.join(class_dir,video_file)
            
            try:
                landmarks = extract_hand_and_arm_landmarks(video_path,max_frames)
                
                if len(landmarks) == 0:
                    print(f"警告: {video_path} 没有提取到关键点")
                    continue
                
                X.append(landmarks)
                y.append(class_idx)
                
            except Exception as e:
                 print(f"处理 {video_path} 时出错: {e}")
    
    return np.array(X),np.array(y)



if __name__ == "__main__":
    print("开始提取特征...")
    X, y = extract_all_features(DATA_DIR,  max_frames=MAX_FRAMES)
    print(f"特征提取完成。X形状: {X.shape}, y形状: {y.shape}")
    
    # 保存提取的特征
    np.save('X_features.npy', X)
    np.save('y_labels.npy', y)
    print("特征已保存到文件")