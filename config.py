
# 数据集路径
DATA_DIR = "toy_dataset"

# 类别列表
CLASSES = []

# MediaPipe参数
MAX_FRAMES = 100  # 每个视频最多处理的帧数
TARGET_LENGTH = 50  # 序列的目标长度

# 模型参数
INPUT_SIZE = 126  # 特征维度：MediaPipe Hand(21点) + Pose(手臂部分关键点)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = len(CLASSES)  # 将在运行时更新
DROPOUT = 0.3

# 训练参数
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = "cpu"  

# 路径
CLASS_MAPPING_FILE = "class_mapping.json"
MODEL_SAVE_PATH = "sign_language_model.pth"