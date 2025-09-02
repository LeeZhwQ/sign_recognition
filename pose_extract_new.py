import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
import json
import gc
import sys
from config import DATA_DIR, MAX_FRAMES, CLASS_MAPPING_FILE

# Windows MediaPipe 修复
if sys.platform.startswith('win'):
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

def extract_hand_and_arm_landmarks(video_path, max_frames):
    """提取手部和手臂关键点 - Windows兼容版本"""
    print(f"  开始处理视频: {video_path}")
    
    landmarks_sequence = []
    
    try:
        # 使用更保守的配置
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        
        # 初始化MediaPipe - 每次处理一个视频时重新初始化
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # 提高置信度
            min_tracking_confidence=0.5,
            model_complexity=0  # 使用更简单的模型
        )
        
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 使用最简单的模型
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  无法打开视频: {video_path}")
            return np.array([])
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = min(total_frames, max_frames)
        print(f"  视频总帧数: {total_frames}, 将处理: {frames_to_process} 帧")
        
        frame_count = 0
        processed_frames = 0
        
        # 使用简单的进度条
        print(f"  处理进度: ", end="")
        progress_interval = max(1, frames_to_process // 10)
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 调整图像大小以提高处理速度
                if frame.shape[0] > 480 or frame.shape[1] > 480:
                    scale = 480 / max(frame.shape[0], frame.shape[1])
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # 图像转换为RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # 处理手部和姿态
                hand_result = hands.process(image)
                pose_result = pose.process(image)
                
                # 提取特征
                feature = []
                
                # 初始化手部特征 (21个关键点 × 2坐标 = 42个值)
                left_hand_features = [0.0] * 42
                right_hand_features = [0.0] * 42
                
                if hand_result.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                        hand_label = handedness.classification[0].label
                        hand_features = []
                        
                        for landmark in hand_landmarks.landmark:
                            hand_features.extend([float(landmark.x), float(landmark.y)])
                        
                        if hand_label == "Left":
                            left_hand_features = hand_features
                        elif hand_label == "Right":
                            right_hand_features = hand_features
                
                # 添加手部特征
                feature.extend(left_hand_features)
                feature.extend(right_hand_features)
                
                # 提取手臂关键点 (6个关键点 × 2坐标 = 12个值)
                if pose_result.pose_landmarks:
                    arm_indices = [11, 13, 15, 12, 14, 16]  # 左肩,左肘,左腕,右肩,右肘,右腕
                    for idx in arm_indices:
                        if idx < len(pose_result.pose_landmarks.landmark):
                            landmark = pose_result.pose_landmarks.landmark[idx]
                            feature.extend([float(landmark.x), float(landmark.y)])
                        else:
                            feature.extend([0.0, 0.0])
                else:
                    feature.extend([0.0] * 12)
                
                landmarks_sequence.append(feature)
                processed_frames += 1
                
                # 显示进度
                if frame_count % progress_interval == 0:
                    print("█", end="", flush=True)
                
            except Exception as e:
                print(f"\n  处理第 {frame_count} 帧时出错: {e}")
                # 继续处理下一帧而不是停止
                feature = [0.0] * (42 + 42 + 12)  # 填充零值
                landmarks_sequence.append(feature)
            
            frame_count += 1
        
        print(f" 完成")
        print(f"  成功处理帧数: {processed_frames}/{frame_count}")
        
    except Exception as e:
        print(f"  视频处理失败: {e}")
        return np.array([])
    
    finally:
        # 确保资源被正确释放
        try:
            if 'cap' in locals():
                cap.release()
            if 'hands' in locals():
                hands.close()
            if 'pose' in locals():
                pose.close()
        except:
            pass
        
        # 强制垃圾回收
        gc.collect()
    
    if len(landmarks_sequence) == 0:
        print(f"  警告: 没有提取到任何特征")
        return np.array([])
    
    result = np.array(landmarks_sequence)
    print(f"  提取特征形状: {result.shape}")
    return result

def extract_all_features(datadir=DATA_DIR, max_frames=MAX_FRAMES):
    """提取所有视频的特征"""
    
    # 加载类别映射
    class_mapping_path = CLASS_MAPPING_FILE if 'CLASS_MAPPING_FILE' in globals() else "./class_mapping.json"
    
    try:
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
        class_to_idx = class_mapping['class_to_idx']
        print(f"加载类别映射: {list(class_to_idx.keys())}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {class_mapping_path}")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"错误: 读取映射文件失败 {e}")
        return np.array([]), np.array([])
    
    X = []
    y = []
    
    # 获取所有类别目录
    try:
        class_dirs = [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))]
        print(f"找到类别目录: {class_dirs}")
    except Exception as e:
        print(f"错误: 无法读取数据目录 {datadir}: {e}")
        return np.array([]), np.array([])
    
    total_videos_processed = 0
    
    for class_name in class_dirs:
        if class_name not in class_to_idx:
            print(f"跳过未映射的类别: {class_name}")
            continue
            
        class_dir = os.path.join(datadir, class_name)
        class_idx = class_to_idx[class_name]
        
        print(f"\n{'='*50}")
        print(f"处理类别: {class_name} (索引: {class_idx})")
        print(f"{'='*50}")
        
        try:
            # 获取视频文件
            all_files = os.listdir(class_dir)
            video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv'))]
            
            print(f"找到 {len(video_files)} 个视频文件")
            if len(video_files) == 0:
                print("没有找到视频文件，跳过此类别")
                continue
                
        except Exception as e:
            print(f"读取类别目录失败: {e}")
            continue
        
        class_success_count = 0
        
        for i, video_file in enumerate(video_files):
            print(f"\n[{i+1}/{len(video_files)}] {video_file}")
            video_path = os.path.join(class_dir, video_file)
            
            try:
                landmarks = extract_hand_and_arm_landmarks(video_path, max_frames)
                
                if len(landmarks) > 0:
                    X.append(landmarks)
                    y.append(class_idx)
                    class_success_count += 1
                    total_videos_processed += 1
                    print(f"  ✓ 成功处理，特征形状: {landmarks.shape}")
                else:
                    print(f"  ✗ 未能提取到特征")
                    
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
            
            # 每处理5个视频后进行一次垃圾回收
            if (i + 1) % 5 == 0:
                gc.collect()
        
        print(f"\n类别 {class_name} 处理完成: {class_success_count}/{len(video_files)} 个视频成功")
    
    print(f"\n{'='*60}")
    print(f"特征提取总结:")
    print(f"总计成功处理: {total_videos_processed} 个视频")
    
    if len(X) == 0:
        print("没有成功处理任何视频！")
        return np.array([]), np.array([])
    
    # 转换为numpy数组
    try:
        X_array = np.array(X, dtype=object)  # 使用object类型以处理不同长度的序列
        y_array = np.array(y)
        print(f"最终数组形状: X={len(X_array)}, y={y_array.shape}")
        return X_array, y_array
    except Exception as e:
        print(f"转换为numpy数组时出错: {e}")
        return np.array([]), np.array([])

if __name__ == "__main__":
    print("MediaPipe 手势识别特征提取")
    print(f"Python版本: {sys.version}")
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"MediaPipe版本: {mp.__version__}")
    print(f"数据目录: {DATA_DIR}")
    print(f"最大帧数: {MAX_FRAMES}")
    print("="*60)
    
    X, y = extract_all_features(DATA_DIR, max_frames=MAX_FRAMES)
    
    if len(X) > 0:
        print(f"\n🎉 特征提取成功完成!")
        print(f"特征数组: {len(X)} 个视频")
        print(f"标签数组: {y.shape}")
        
        # 保存特征
        try:
            np.save('X_features.npy', X)
            np.save('y_labels.npy', y)
            print("特征已保存到 X_features.npy 和 y_labels.npy")
        except Exception as e:
            print(f"保存文件时出错: {e}")
    else:
        print("\n❌ 没有成功提取任何特征")
        print("\n可能的解决方案:")
        print("1. 检查视频文件是否损坏")
        print("2. 尝试转换视频格式 (推荐MP4)")
        print("3. 确保视频中有清晰的手部动作")
        print("4. 检查MediaPipe安装是否正确")
