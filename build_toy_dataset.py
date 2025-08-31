import json
import os
import shutil
import argparse

def create_toy_dataset(json_path, video_dir, output_dir, selected_glosses):
    """创建WLASL数据集的子集（从本地视频文件复制）"""
    
    # 确保路径是绝对路径
    json_path = os.path.abspath(json_path)
    video_dir = os.path.abspath(video_dir)
    output_dir = os.path.abspath(output_dir)
    
    print(f"JSON路径: {json_path}")
    print(f"视频目录: {video_dir}")
    print(f"输出目录: {output_dir}")
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个选定的类别创建文件夹
    for gloss in selected_glosses:
        gloss_dir = os.path.join(output_dir, gloss)
        os.makedirs(gloss_dir, exist_ok=True)
    
    # 遍历数据并收集选定类别的视频
    copied_count = 0
    skipped_count = 0
    
    print("开始处理数据...")
    for i, entry in enumerate(data):
        if i % 10 == 0:  # 每处理10个条目打印一次进度
            print(f"已处理 {i}/{len(data)} 个条目...")
            
        gloss = entry.get("gloss", "")
        
        if gloss in selected_glosses:
            instances = entry.get("instances", [])
            
            for instance in instances:
                video_id = instance.get("video_id", "")
                
                if not video_id:
                    skipped_count += 1
                    continue
                
                # 构建源视频文件路径 - 尝试不同的格式
                possible_paths = [
                    os.path.join(video_dir, f"{video_id}.mp4"),
                    os.path.join(video_dir, f"{int(video_id):05d}.mp4"),  # 尝试5位数字格式
                ]
                
                source_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        source_path = path
                        break
                
                # 检查源文件是否存在
                if not source_path:
                    print(f"源文件不存在: {video_id}.mp4 (在目录 {video_dir})")
                    skipped_count += 1
                    continue
                
                # 创建目标文件名
                filename = f"{video_id}.mp4"
                dest_path = os.path.join(output_dir, gloss, filename)
                
                # 如果文件已存在，跳过复制
                if os.path.exists(dest_path):
                    skipped_count += 1
                    continue
                
                # 复制视频文件
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    print(f"已复制: {source_path} -> {dest_path}")
                except Exception as e:
                    print(f"复制视频失败: {source_path}, 错误: {e}")
                    skipped_count += 1
    
    print(f"处理完成! 成功复制: {copied_count}, 跳过: {skipped_count}")

# 如果您不想使用命令行参数，可以直接在这里设置路径
if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置相对路径，基于脚本所在目录
    json_path = os.path.join(script_dir, "dataset", "WLASL_v0.3.json")
    video_dir = os.path.join(script_dir, "dataset", "videos", "videos")
    output_dir = os.path.join(script_dir, "toy_dataset")
    glosses = ["book","mean","most","music","new","none","office","order","pants","party"]
    
    create_toy_dataset(json_path, video_dir, output_dir, glosses)