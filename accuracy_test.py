from tqdm import tqdm
from model import GestureLSTM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class GestureTrainer:
    
    def __init__(self,model,device,save_dir = "./checkpoints"):
        
        self.model = model
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.train_accs = []
        
    
    def train_epoch(self,train_loader,optimizer,criterion):
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc= "Training")
        
        for batch_idx, (data,targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            #前向传播
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs,targets)
            
            #反向
            loss.backward()
            optimizer.step()
            
            #统计
            total_loss += loss.item()
            _,predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
             # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.* correct / total
            
        return avg_loss,accuracy
    
    
    def save_checkpoint(self, epoch, accuracy, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    
        
    def train(self,train_loader,test_loader=None,epochs=50,lr=0.01,weight_decay=1e-4,save_every=10):
        
        #优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        #学习率调整
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode= "min",
            factor=0.5,
            patience=10
        )
        
        best_train_acc = 0.0
        
        print(f"开始训练，共{epochs}个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 80)
        
        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            
            train_loss,train_acc = self.train_epoch(train_loader=train_loader,optimizer=optimizer,criterion=criterion)
            
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            scheduler.step(train_loss)
            
             # 打印结果
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                self.save_checkpoint(epoch,train_acc,"best_model.pth")
                print(f"保存最佳模型 (训练准确率: {train_acc:.2f}%)")
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch,train_acc,f'checkpoint_epoch_{epoch+1}.pth')
                print(f"保存检查点: checkpoint_epoch_{epoch+1}.pth")
            
            print("-" * 80)
        
        print(f"训练完成！最佳训练准确率: {best_train_acc:.2f}%")
        return best_train_acc
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        return checkpoint['epoch'], checkpoint['accuracy']
    
    
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.set_title('Training Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accs, label='Training Accuracy', color='blue')
        ax2.set_title('Training Accuracy Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self,model,test_loader,device,class_names = os.listdir("toy_dataset")):
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data,targets in tqdm(test_loader, desc='Testing'):
                data,targets = data.to(device),targets.to(device)
                output = model(data)
                _,predicted = torch.max(output,1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        
        accuracy = accuracy_score(all_targets,all_predictions)
        print(f"测试准确率: {accuracy:.4f}")
        
        print("\n分类报告:")
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        return accuracy,all_predictions,all_targets
    
    
def load_data():
    try:
        print("加载预处理后的数据...")
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
            
        print(f"数据加载成功:")
        print(f"  训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
            
            # 转换为tensor
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
            
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
        
    except FileNotFoundError as e:
        print(f"❌ 找不到预处理数据文件: {e}")
        print("请先运行数据预处理脚本生成必要的文件")
        raise
    except Exception as e:
        print(f"❌ 数据加载出错: {e}")
        raise
    
    
if __name__ == "__main__":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        X_train,X_test,y_train,y_test = load_data()
        seq_length = X_train.shape[1]  # 序列长度
        input_dim = X_train.shape[2]   # 输入维度 (应该是96)
        num_classes = len(torch.unique(y_train))  # 类别数量
        
        print(f"数据维度信息:")
        print(f"  序列长度: {seq_length}")
        print(f"  输入维度: {input_dim}")
        print(f"  类别数量: {num_classes}")
        
        
        batch_size = 32
        train_dataset = TensorDataset(X_train , y_train)
        test_dataset = TensorDataset(X_test,y_test)
        
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
        
        print(f"数据加载器创建成功:")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  测试批次数: {len(test_loader)}")
        
        model = GestureLSTM()
        
        trainer = GestureTrainer(model,device=device)
        
        # 加载最佳模型进行最终测试
        print("\n使用最佳模型进行最终测试...")
        trainer.load_checkpoint('best_model.pth')
        
        test_accuracy,predictions,targets = trainer.evaluate_model(model,test_loader,device=device)
        
        print(f" 最终测试准确率: {test_accuracy*100:.2f}%")
        
        
        # 保存最终结果
        results = {
            'final_test_acc': test_accuracy * 100,
            'num_classes': num_classes,
            'model_params': sum(p.numel() for p in model.parameters()),
            'seq_length': seq_length,
            'input_dim': input_dim
        }
        
        import json
        with open('./checkpoints/training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: training_results.json")