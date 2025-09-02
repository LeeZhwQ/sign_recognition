import torch
from train import GestureLSTM
from accuracy_test import GestureTrainer
import numpy as np
import gc
import cv2
import mediapipe as mp
import config
from pose_extract_new import extract_hand_and_arm_landmarks
from preprocessing import adjust_sequence_length,normalize_landmarks
import json


x = extract_hand_and_arm_landmarks("./toy_dataset/book/70266.mp4",max_frames=config.MAX_FRAMES)

normolized_x = []

for frame_id,frame in enumerate(x):
    try:
        normolized_frame = normalize_landmarks(frame)
        normolized_x.append(normolized_frame)
    except Exception as e:
        print(f"警告: 第 {frame_id} 帧处理失败: {e}")
         #使用零帧作为兜底
        normolized_x.append(np.zeros(96, dtype=np.float32))
                    
normalized_sequence = np.array(normolized_x)
processed_sequence = adjust_sequence_length(normalized_sequence,config.TARGET_LENGTH)

print("特征预处理完成，准备加载模型…")
model = GestureLSTM()
print("模型实例化完成")
trainer = GestureTrainer(model, device=config.DEVICE)
print("Trainer 实例化完成")
trainer.load_checkpoint('best_model.pth')
print("权重加载完成")
x = torch.tensor(processed_sequence, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
print("张量构造完成", x.shape, x.device)

x = x.to(config.DEVICE)

model.eval()
try:
    with torch.no_grad():
        logits = trainer.model(x)
        _, predicted = torch.max(logits, 1)
        print("预测索引:", predicted.item())
except Exception as e:
    print("推理出错:", e)
    import traceback
    traceback.print_exc()

idx_to_class = []

with open("./class_mapping.json", 'r') as f:
    mapping = json.load(f)
    idx_to_class = mapping['idx_to_class']


print("预测结果：", idx_to_class[str(predicted.item())])
