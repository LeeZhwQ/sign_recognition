import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TARGET_LENGTH
import joblib

def normalize_landmarks(landmarks):
    
    '''以肩膀中心点进行归一化，去除噪声'''
    landmarks = np.array(landmarks,dtype=np.float32)
    
    if len(landmarks != 96):
        if len(landmarks) < 96:
            landmarks = np.pad(landmarks,(0,96-len(landmarks)),'constant',constant_values=0.0)
        else:
            landmarks = landmarks[:96]
    
    landmarks = landmarks.reshape(48,2)
    
    left_shoulder_index = 42
    right_shoulder_index = 45
    
    left_shoulder = landmarks[left_shoulder_index]
    right_shoulder = landmarks[right_shoulder_index]
    
    if np.any(left_shoulder != 0) and np.any(right_shoulder != 0):
        center = (left_shoulder + right_shoulder) / 2
        scale = np.linalg.norm(right_shoulder - left_shoulder)
    elif np.any(left_shoulder != 0):
        # 只有左肩有效
        center = left_shoulder
        scale = 1.0
    elif np.any(right_shoulder != 0):
        # 只有右肩有效
        center = right_shoulder
        scale = 1.0
    else:
        non_zero = np.any(landmarks != 0 , axis= 1)
        if (np.any(non_zero)):
            center = np.mean(landmarks[non_zero],axis=0)
        else:
            center = np.array([0.5,0.5],dtype= np.float32)
        scale = 1.0
    
    normalized = landmarks - center
    if scale > 1e-6:
        normalized = normalized / scale
    
    result  = normalized.flatten()
    assert len(result) == 96, f"归一化后维度错误: {len(result)}, 期望96"
    
    return result


def adjust_sequence_length(sequence,target_length):
    
    sequence = np.array(sequence)
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    elif current_length > target_length:
        start = (current_length - target_length) // 2
        return sequence[start:start + target_length]
    else:
        total_padding_length = target_length - current_length
        
        left_padding_length = total_padding_length // 2
        right_padding_length = total_padding_length - left_padding_length
        
        if left_padding_length > 0:
            first_frame = sequence[0:1]
            left_pad = np.repeat(first_frame,left_padding_length,axis=0)
        else:
            left_pad = np.empty((0, sequence.shape[1]))
            
        if right_padding_length > 0:
            last_frame = sequence[-1:]
            right_pad = np.repeat(last_frame,right_padding_length,axis=0)
        else:
            right_pad = np.empty((0, sequence.shape[1]))
            
        result = np.vstack([left_pad,sequence,right_pad])
        
    assert len(result) == target_length, f"序列长度调整错误: {len(result)}, 期望{target_length}"
    
    return result

def preprocess_data(X,y,target_length = TARGET_LENGTH):
    '''主处理函数'''
    print(f"开始预处理 {len(X)} 个序列...")
    print(f"目标序列长度: {target_length}")
    
    processed_X = []
    valid_indices = []
    failed_count = 0
    
    for idx, sequence in enumerate(X):
        try:
            # 跳过空序列
            if len(sequence) == 0:
                print(f"警告: 序列 {idx} 为空，跳过")
                failed_count += 1
                continue
            
            normalized_sequence = []
            for frame_id,frame in enumerate(sequence):
                try:
                    normolized_frame = normalize_landmarks(frame)
                    normalized_sequence.append(normolized_frame)
                except Exception as e:
                    print(f"警告: 序列 {idx} 第 {frame_id} 帧处理失败: {e}")
                    # 使用零帧作为兜底
                    normalized_sequence.append(np.zeros(96, dtype=np.float32))
                    
            normalized_sequence = np.array(normalized_sequence)
            processed_sequence = adjust_sequence_length(normalized_sequence,target_length)
            
            assert processed_sequence.shape == (target_length, 96), \
                f"序列 {idx} 最终形状错误: {processed_sequence.shape}, 期望({target_length}, 96)"
            
            processed_X.append(processed_sequence)
            valid_indices.append(idx)
        
        except Exception as e:
            print(f"错误: 序列 {idx} 处理完全失败: {e}")
            failed_count += 1
            continue
        
        # 进度显示
        if (idx + 1) % 100 == 0:
            print(f"已处理: {idx + 1}/{len(X)}, 成功: {len(processed_X)}, 失败: {failed_count}")
    
    
    processed_X = np.array(processed_X,dtype=np.float32)
    processed_y = y[valid_indices]
    
    print(f"预处理完成:")
    print(f"  成功处理: {len(processed_X)} 个序列")
    print(f"  失败: {failed_count} 个序列")
    print(f"  最终形状: X={processed_X.shape}, y={processed_y.shape}")
    
    # 最终形状验证
    expected_shape = (len(processed_X), target_length, 96)
    assert processed_X.shape == expected_shape, \
        f"最终X形状错误: {processed_X.shape}, 期望{expected_shape}"
    
    return processed_X, processed_y


def split_and_normalize(X,y):
    """数据划分和标准化"""
    print(f"开始数据划分...")
    print(f"输入数据形状: X={X.shape}, y={y.shape}")
    
    # 检查类别分布
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"类别分布: {dict(zip(unique_classes, class_counts))}")
    
    
        
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.1,random_state=42, stratify=y
    )
        
    
    print(f"数据划分完成:")
    print(f"  训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
    
    print("应用全局标准化...")
    scaler = StandardScaler()
    
    original_train_shape = X_train.shape
    original_test_shape = X_test.shape
    
    X_train_flat = X_train.reshape(-1,96)
    X_test_flat = X_test.reshape(-1,96)
    
    X_train_norm_flat = scaler.fit_transform(X_train_flat)
    X_test_norm_flat = scaler.transform(X_test_flat)
    
    X_train_norm = X_train_norm_flat.reshape(original_train_shape)
    X_test_norm = X_test_norm_flat.reshape(original_test_shape)
    
    print("标准化完成")
    
    return (X_train_norm,X_test_norm), (y_train,y_test), scaler


if __name__ == "__main__":
    print("手语识别数据预处理")
    print("=" * 50)
    
    try:
        # 加载原始数据
        print("加载原始数据...")
        X = np.load('X_features.npy', allow_pickle=True)
        y = np.load('y_labels.npy')
        print(f"原始数据: X包含{len(X)}个序列, y形状{y.shape}")
        
        # 数据完整性检查
        if len(X) != len(y):
            raise ValueError(f"特征和标签数量不匹配: {len(X)} vs {len(y)}")
        
        # 主要预处理
        X_processed, y_processed = preprocess_data(X, y, target_length=TARGET_LENGTH)
        
        # 数据划分和标准化
        (X_train,  X_test), (y_train, y_test), scaler = split_and_normalize(X_processed, y_processed)
        
        # 保存处理后的数据
        print("\n保存处理后的数据...")
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        joblib.dump(scaler, 'feature_scaler.pkl')
        
        # 最终报告
        print("\n✅ 数据预处理完成!")
        print(f"📊 最终数据集:")
        print(f"   训练集: {X_train.shape} -> 每个视频 {TARGET_LENGTH}×96")
        print(f"   测试集: {X_test.shape} -> 每个视频 {TARGET_LENGTH}×96")
        print(f"💾 保存的文件:")
        print(f"   - X_train.npy, X_test.npy")
        print(f"   - y_train.npy, y_test.npy")
        print(f"   - feature_scaler.pkl")
        
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e}")
        print("请确保已运行特征提取脚本生成 X_features.npy 和 y_labels.npy")
        
    except Exception as e:
        print(f"❌ 处理过程出错: {e}")
        import traceback
        traceback.print_exc()
    
        