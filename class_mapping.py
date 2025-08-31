import os
import json
from config import DATA_DIR,CLASS_MAPPING_FILE

def get_class_mapping():
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))]
    class_to_idx = {cls:i for i,cls in enumerate(classes)}
    idx_to_class = {i:cls for cls,i in class_to_idx.items()}
    return class_to_idx,idx_to_class,classes

def save_class_mapping(class_to_idx,idx_to_class,file_path):
    with open(file_path,'w') as f:
        json.dump({'class_to_idx':class_to_idx,'idx_to_class':idx_to_class},f)
    
def load_class_mapping(file_path):
    """从文件加载类别映射"""
    with open(file_path, 'r') as f:
        mapping = json.load(f)
    return mapping['class_to_idx'], mapping['idx_to_class']

if __name__ == "__main__":
    class_to_idx,idx_to_class,classes = get_class_mapping()
    save_class_mapping(class_to_idx,idx_to_class,CLASS_MAPPING_FILE)
    print(f"找到 {len(classes)} 个类别: {classes}")
    print("类别映射已保存到", CLASS_MAPPING_FILE)
    