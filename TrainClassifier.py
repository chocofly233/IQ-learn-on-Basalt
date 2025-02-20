
import swanlab
swanlab.login(api_key="nDJSrz7etgAUqLhLjreeE")
run = swanlab.init(
    # 设置项目
    project="Kaggle_CaveandBC_training",
    # 跟踪超参数与实验元数据
    config={
        
        "epochs": 100,
    },
)
import sys
sys.path.append('../input')

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch import functional as F
import numpy as np
import shutil
from PIL import Image
import torchvision.transforms as T
import tqdm

# 添加动作键映射常量
ACTION_KEYS = [
    'forward',  # W
    'back',     # S
    'left',     # A
    'right',    # D
    'jump',     # Space
    'ESC'
]
KEY_MAPPING = {
    'key.keyboard.w': 'forward',
    'key.keyboard.s': 'back',
    'key.keyboard.a': 'left',
    'key.keyboard.d': 'right', 
    'key.keyboard.space': 'jump',
    'key.keyboard.escape': 'ESC'  # 添加ESC键映射
}

# Mouse转Camera的转换系数 (这些值需要根据实际游戏设置调整)
MOUSE_TO_CAMERA_X = 0.15  # 水平视角转换系数
MOUSE_TO_CAMERA_Y = 0.15  # 垂直视角转换系数

# 添加新的常量定义
MAX_GRAD_NORM = 5.0
LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
class CompressedCaveDataset(Dataset):
    def __init__(self, cave_dir='/kaggle/input/pretrain/compressed_cave/compressed_cave', non_cave_dir='/kaggle/input/pretrain/compressed_non_cave/compressed_non_cave'):
        self.cave_batches = sorted([os.path.join(cave_dir, f) for f in os.listdir(cave_dir) if f.endswith('.pth')])
        self.non_cave_batches = sorted([os.path.join(non_cave_dir, f) for f in os.listdir(non_cave_dir) if f.endswith('.pth')])
        
        # 缓存每个batch的实际大小
        self.cave_batch_sizes = []
        self.non_cave_batch_sizes = []
        # 计算总样本数和每个batch的大小
        total_cave_samples = 0
        total_non_cave_samples = 0
        
        for batch_file in self.cave_batches:
            batch = torch.load(batch_file, map_location='cpu')
            batch_size = batch.size(0)
            self.cave_batch_sizes.append(batch_size)
            total_cave_samples += batch_size
            
        for batch_file in self.non_cave_batches:
            batch = torch.load(batch_file, map_location='cpu')
            batch_size = batch.size(0)
            self.non_cave_batch_sizes.append(batch_size)
            total_non_cave_samples += batch_size
        
        self.cave_samples = total_cave_samples
        self.non_cave_samples = total_non_cave_samples
        
        # 生成标签
        self.labels = torch.cat([
            torch.ones(self.cave_samples),
            torch.zeros(self.non_cave_samples)
        ])
        
        # 创建累积大小数组，用于快速定位
        self.cave_cumsum = np.cumsum([0] + self.cave_batch_sizes)
        self.non_cave_cumsum = np.cumsum([0] + self.non_cave_batch_sizes)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if idx < self.cave_samples:
            # 在cave样本中查找
            batch_idx = np.searchsorted(self.cave_cumsum, idx, side='right') - 1
            sample_idx = idx - self.cave_cumsum[batch_idx]
            batch = torch.load(self.cave_batches[batch_idx], map_location='cpu')
            return batch[sample_idx], self.labels[idx]
        else:
            # 在non-cave样本中查找
            adj_idx = idx - self.cave_samples
            batch_idx = np.searchsorted(self.non_cave_cumsum, adj_idx, side='right') - 1
            sample_idx = adj_idx - self.non_cave_cumsum[batch_idx]
            batch = torch.load(self.non_cave_batches[batch_idx], map_location='cpu')
            return batch[sample_idx], self.labels[idx]
def load_model_weights(model, weights_path, device='cuda'):
    """加载模型权重的通用函数"""
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        # 处理不同格式的checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded weights from {weights_path}")
        return True
    except Exception as e:
        print(f"Failed to load weights from {weights_path}: {e}")
        return False
class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 添加打印尺寸的测试代码
        test_input = torch.zeros(1, 3, 72, 128)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),     # -> [32,35,63]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),    # -> [64,16,30]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),    # -> [64,14,28]
            nn.ReLU(),
            nn.Flatten()                        # -> [25088]
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            test_output = self.cnn(test_input)
            feature_dim = test_output.shape[1]
            print(f"CNN输出维度: {feature_dim}")
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),  # 使用实际计算的feature_dim
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        # 确保输入维度正确 [B,H,W,C] -> [B,C,H,W]
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)
        return self.classifier(features)
# 行为克隆模型
class BehaviorCloneModel(nn.Module):
    def __init__(self):
        super(BehaviorCloneModel, self).__init__()
        # 使用FindCave的CNN结构
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            test_input = torch.zeros(1, 3, 72, 128)
            test_output = self.cnn(test_input)
            n_flatten = test_output.shape[1]
        
        # 改进的动作预测头
        self.camera_head = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh()
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(ACTION_KEYS)),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.cnn(x)
        camera_actions = self.camera_head(features)
        discrete_actions = self.action_head(features)
        return camera_actions, discrete_actions
    def load_weights(self, weights_path, device='cuda'):
        """加载BC模型权重"""
        return load_model_weights(self, weights_path, device)
class MinecraftDatasetFromJSONL(Dataset):
    def __init__(self):
        self.jsonl_dir = '/kaggle/input/new-processed-jsons/processed_jsons'
        self.compressed_dir = '/kaggle/input/frameticks/compressed_frames_with_ticks'
        
        # 加载压缩索引
        try:
            with open(os.path.join(self.compressed_dir, 'compression_index.json'), 'r') as f:
                self.compression_info = json.load(f)
            print(f"Successfully loaded compression index with {self.compression_info['total_frames']} frames")
        except FileNotFoundError:
            print("Warning: No compression index found, reverting to sequential mapping")
            self.compression_info = None
        
        self.data = []
        self.batch_tensors = {}
        
        # 读取JSONL文件
        jsonl_files = sorted([f for f in os.listdir(self.jsonl_dir) if f.endswith('.jsonl')])
        for json_file in jsonl_files:
            video_id = os.path.splitext(json_file)[0]  # 从文件名获取video_id
            with open(os.path.join(self.jsonl_dir, json_file), 'r') as f:
                for line in f:
                    if line.strip():
                        frame_data = json.loads(line)
                        # 转换数据格式
                        action_data = {
                            'video_id': video_id,
                            'tick': frame_data['tick']+1,
                            'mouse_dx': frame_data['mouse']['dx'],
                            'mouse_dy': frame_data['mouse']['dy'],
                            'keys': frame_data['keyboard']['keys']
                        }
                        self.data.append(action_data)
        
        self.samples_per_batch = self.compression_info['batch_size'] if self.compression_info else 32
        print(f"加载了 {len(self.data)} 条动作数据")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        action_data = self.data[idx]
        
        # 使用压缩索引找到对应的图像
        if self.compression_info:
            # 尝试两种可能的文件名格式
            frame_name_simple = f"frame_{action_data['tick']:04d}.png"  # 简单格式
            frame_name_full = f"{action_data['video_id']}_frame_{action_data['tick']:04d}.png"  # 完整格式
            
            # 检查哪个文件名存在于索引中
            if frame_name_simple in self.compression_info['frame_mapping']:
                mapping = self.compression_info['frame_mapping'][frame_name_simple]
            elif frame_name_full in self.compression_info['frame_mapping']:
                mapping = self.compression_info['frame_mapping'][frame_name_full]
            else:
                print(f"Warning: Neither {frame_name_simple} nor {frame_name_full} found in index")
                mapping = {
                    'batch_idx': idx // self.samples_per_batch,
                    'sample_idx': idx % self.samples_per_batch
                }
            
            batch_idx = mapping['batch_idx']
            sample_idx = mapping['sample_idx']
        else:
            batch_idx = idx // self.samples_per_batch
            sample_idx = idx % self.samples_per_batch
        
        # 加载图像数据
        if batch_idx not in self.batch_tensors:
            batch_path = os.path.join(self.compressed_dir, f'batch_{batch_idx}.pth')
            try:
                self.batch_tensors[batch_idx] = torch.load(batch_path, map_location='cpu')
            except Exception as e:
                print(f"Error loading batch {batch_idx}: {e}")
                return torch.zeros(3, 72, 128), torch.zeros(2), torch.zeros(len(ACTION_KEYS))
        
        try:
            image = self.batch_tensors[batch_idx][sample_idx]
        except Exception as e:
            print(f"Error accessing image at batch {batch_idx}, sample {sample_idx}: {e}")
            return torch.zeros(3, 72, 128), torch.zeros(2), torch.zeros(len(ACTION_KEYS))
        
        # 处理动作标签
        camera_actions = torch.tensor([
            action_data["mouse_dx"] * MOUSE_TO_CAMERA_X,
            action_data["mouse_dy"] * MOUSE_TO_CAMERA_Y
        ], dtype=torch.float32)
        
        discrete_actions = torch.zeros(len(ACTION_KEYS))
        for key in action_data['keys']:
            if key in KEY_MAPPING:
                action_idx = ACTION_KEYS.index(KEY_MAPPING[key])
                discrete_actions[action_idx] = 1.0
        
        return image, camera_actions, discrete_actions

# 添加自定义的collate函数
def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    mouse_actions = torch.stack([item[1] for item in batch])
    keyboard_actions = torch.stack([item[2] for item in batch])
    return images, mouse_actions, keyboard_actions
def evaluate_cave_detector(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.long()).sum().item()
            total += target.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def train_cave_detector(device='cuda'):
    # 数据处理
    # 创建数据集
    dataset = CompressedCaveDataset()
    
    # 划分数据集
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"总样本数: {len(dataset)}")
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 初始化模型并移至指定设备
    cave_detector = DiscriminatorModel().to(device)
    optimizer = optim.Adam(cave_detector.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    best_val_accuracy = 0
    best_model_state = None
    steps_per_epoch = len(train_loader)
    
    for epoch in range(100):
        cave_detector.train()
        epoch_loss = 0
        
        # 训练循环
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = cave_detector(data)
            loss = criterion(output, target.long())
            swanlab.log({
                "cave_detector_loss": loss.item()
            })
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 每50个batch打印一次进度
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch}/100] Batch [{batch_idx}/{steps_per_epoch}] '
                      f'Loss: {epoch_loss/(batch_idx+1):.4f}')
        
        # 验证阶段
        cave_detector.eval()  # 切换到评估模式
        val_loss, val_accuracy = evaluate_cave_detector(
            cave_detector, val_loader, criterion, device
        )
        
        print(f'\nEpoch {epoch} 结果:')
        print(f'平均训练损失: {epoch_loss/steps_per_epoch:.4f}')
        print(f'验证损失: {val_loss:.4f}')
        print(f'验证准确率: {val_accuracy:.2f}%\n')
        swanlab.log({
            "cave_detector_train_loss": epoch_loss / steps_per_epoch,
            "cave_detector_val_loss": val_loss,
            "cave_detector_val_accuracy": val_accuracy
        })
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': cave_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, 'cave_detector_best.pth')

def train_step(model, batch, optimizer, device):
    images, camera_actions, discrete_actions = batch
    images = images.to(device)
    #print(images.shape)
   # images = torch.reshape(batch,3,640,360)
    camera_actions = camera_actions.to(device)
    discrete_actions = discrete_actions.to(device)
    
    # 前向传播
    camera_pred, discrete_pred = model(images)
    
    # 计算损失
    camera_loss = nn.MSELoss()(camera_pred, camera_actions)
    discrete_loss = nn.BCELoss()(discrete_pred, discrete_actions)
    
    # 使用权重平衡两种损失
    total_loss = camera_loss + 0.5 * discrete_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'camera_loss': camera_loss.item(),
        'discrete_loss': discrete_loss.item()
    }

def validate(model, val_loader, device):
    model.eval()
    total_camera_loss = 0
    total_discrete_loss = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images, camera_actions, discrete_actions = [x.to(device) for x in batch]
            camera_pred, discrete_pred = model(images)
            
            camera_loss = nn.MSELoss()(camera_pred, camera_actions)
            discrete_loss = nn.BCELoss()(discrete_pred, discrete_actions)
            loss = camera_loss + 0.5 * discrete_loss
            
            total_camera_loss += camera_loss.item()
            total_discrete_loss += discrete_loss.item()
            total_loss += loss.item()
    
    return {
        'val_camera_loss': total_camera_loss / len(val_loader),
        'val_discrete_loss': total_discrete_loss / len(val_loader),
        'val_total_loss': total_loss / len(val_loader)
    }

def load_pretrained_models(bc_path='best_model.pth', disc_path='cave_detector.pth', device='cuda'):
    """加载预训练的BC模型和判别器"""
    bc_model = BehaviorCloneModel()
    disc_model = DiscriminatorModel(input_dim=3136, hidden_dim=512)
    
    bc_loaded = bc_model.load_weights(bc_path, device)
    disc_loaded = disc_model.load_weights(disc_path, device)
    
    if not bc_loaded or not disc_loaded:
        print("Warning: Some models failed to load")
    
    return bc_model, disc_model

def main():
    print("Starting behavior cloning training...")
    
    # 加载压缩数据集
    print("Loading dataset...")
    dataset = MinecraftDatasetFromJSONL()
    
    # 数据集分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=custom_collate,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=2
    )
    
    print(f"数据集统计:")
    print(f"- 总样本数: {len(dataset)}")
    print(f"- 训练集: {len(train_dataset)}")
    print(f"- 验证集: {len(val_dataset)}")
    
    # 初始化模型和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BehaviorCloneModel().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    best_val_loss = float('inf')
    num_epochs = 100
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 训练阶段
        for batch_idx, batch in enumerate(train_loader):
            # 执行单步训练
            loss_dict = train_step(model, batch, optimizer, device)
            total_loss += loss_dict['total_loss']
            num_batches += 1
            
            if batch_idx % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Batch {batch_idx}, Avg Loss: {avg_loss:.4f}")
                print(f"Camera Loss: {loss_dict['camera_loss']:.4f}")
                print(f"Discrete Loss: {loss_dict['discrete_loss']:.4f}")
        
        # 计算训练epoch的平均损失
        epoch_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} Training Loss: {epoch_loss:.4f}")
        
        # 验证阶段
        val_losses = validate(model, val_loader, device)
        print(f"Validation Results:")
        print(f"- Camera Loss: {val_losses['val_camera_loss']:.4f}")
        print(f"- Discrete Loss: {val_losses['val_discrete_loss']:.4f}")
        print(f"- Total Loss: {val_losses['val_total_loss']:.4f}")
        
        # 记录损失
        swanlab.log({
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss),
            "val_camera_loss": float(val_losses['val_camera_loss']),
            "val_discrete_loss": float(val_losses['val_discrete_loss']),
            "val_total_loss": float(val_losses['val_total_loss'])
        })
        
        # 保存最佳模型
        if val_losses['val_total_loss'] < best_val_loss:
            best_val_loss = val_losses['val_total_loss']
            print(f"保存新的最佳模型 (val_loss: {best_val_loss:.4f})")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model.pth')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
if __name__ == "__main__":
    main()