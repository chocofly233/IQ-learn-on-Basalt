import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import math
from argparse import ArgumentParser
import torchvision.transforms as T
import cv2
import swanlab
import pickle
import gym
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt
import matplotlib
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from IQLAgent import IQLearnAgent     # 独立训练的 IQ-Learn 模型
from agent import MineRLAgent, ENV_KWARGS   # 预训练的 VPT 模型
from Video_data_loader import json_action_to_env_action
from Video_data_loader import DataLoader as DT
from path_tracker import PathTracker
import json
import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
matplotlib.use('Agg')  # 使用非交互式后端
RESOLUTION = (72,128)
DISCRETE_ACTIONS = [
    {'forward': 1},        # Move: Forward
    {'back': 1},           # Move: Back
    {'left': 1},           # Move: Left
    {'right': 1},          # Move: Right
    {'camera': [10, 0]},   # Look: Up
    {'camera': [-10, 0]},  # Look: Down
    {'camera': [0, -10]},  # Look: Left
    {'camera': [0, 10]},   # Look: Right
    {'forward': 1, 'jump': 1},  # Forward Jump
    {'jump': 1},                # Jump
    {'attack': 1}               # Attack
]

NUM_DISCRETE = len(DISCRETE_ACTIONS)
def preprocess_expert_data(expert_frames, expert_actions, device='cuda', cave_detector=None):
    """
    综合处理专家数据函数，包含:
    1. 图像预处理和数据增强
    2. 动作转换为one-hot向量
    3. 矿洞特征检测
    4. DrQ风格增强

    Args:
        expert_frames: 专家视频帧列表
        expert_actions: 对应的专家动作列表
        device: 计算设备
        cave_detector: 预训练的洞穴检测器模型
    
    Returns:
        处理后的数据字典
    """
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import numpy as np
    from collections import defaultdict
    
    # 基本预处理转换
    base_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    
    # DrQ风格的随机增强
    drq_transforms = [
        T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
            T.ToTensor()
        ]),
        T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.3)
            ], p=0.4),
            T.ToTensor()
        ])
    ]
    
    # 矿洞增强 - 增强暗区域对比度
    def enhance_dark_regions(img_tensor):
        # 计算亮度
        brightness = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
        # 创建暗区掩码
        dark_mask = brightness < 0.3
        # 在暗区域增强对比度
        enhanced = img_tensor.clone()
        for c in range(3):
            channel = enhanced[c]
            dark_pixels = channel[dark_mask]
            if dark_pixels.numel() > 0:
                dark_min = dark_pixels.min()
                dark_max = dark_pixels.max()
                if dark_max > dark_min:
                    # 增强对比度
                    channel[dark_mask] = (channel[dark_mask] - dark_min) / (dark_max - dark_min) * 0.6
        return enhanced

    # 准备数据容器
    processed_data = defaultdict(list)
    n_frames = len(expert_frames)
    
    # 处理每一帧
    for i in range(n_frames - 1):  # 最后一帧没有下一帧
        # 获取当前帧和下一帧
        current_frame = expert_frames[i]
        next_frame = expert_frames[i + 1]
        current_action = expert_actions[i]
        
        # 基本预处理
        current_tensor = base_transform(current_frame).to(device)
        next_tensor = base_transform(next_frame).to(device)
        
        # DrQ风格增强 - 每个帧使用多个增强版本
        augmented_current = [current_tensor]
        augmented_next = [next_tensor]
        
        for transform in drq_transforms:
            augmented_current.append(transform(current_frame).to(device))
            augmented_next.append(transform(next_frame).to(device))
        
        # 矿洞特征增强
        
        for j in range(len(augmented_current)):
            augmented_current[j] = enhance_dark_regions(augmented_current[j])
            augmented_next[j] = enhance_dark_regions(augmented_next[j])
        
        # 随机平移增强 (DrQ核心)
        for j in range(len(augmented_current)):
            augmented_current[j] = random_shift(augmented_current[j].unsqueeze(0), pad=4).squeeze(0)
            augmented_next[j] = random_shift(augmented_next[j].unsqueeze(0), pad=4).squeeze(0)
        
        # 存储增强后的观测
        processed_data['states'].append(torch.stack(augmented_current))
        processed_data['next_states'].append(torch.stack(augmented_next))
        
        # 转换动作为one-hot向量
        action_tensor = unified_convert_action(current_action, device)
        processed_data['actions'].append(action_tensor)
        
        # 检测矿洞概率
        if cave_detector is not None:
            with torch.no_grad():
                # 使用标准转换以匹配cave_detector预期输入
                cave_input = trans_pipeline(current_frame).unsqueeze(0).to(device)
                logits = cave_detector(cave_input)
                probs = F.softmax(logits, dim=1)
                cave_prob = probs[0, 1].item()
                
                next_cave_input = trans_pipeline(next_frame).unsqueeze(0).to(device)
                next_logits = cave_detector(next_cave_input)
                next_probs = F.softmax(next_logits, dim=1)
                next_cave_prob = next_probs[0, 1].item()
        else:
            cave_prob = 0.0
            next_cave_prob = 0.0
            
        processed_data['cave_probs'].append(cave_prob)
        processed_data['next_cave_probs'].append(next_cave_prob)
        
        # 设置专家标记和完成状态
        processed_data['is_expert'].append(True)
        processed_data['dones'].append(False)
        
        # 根据矿洞概率设置奖励 (可选)
        # 这里设置为0，因为IQL应该从数据中学习奖励
        reward = 0.0
        processed_data['rewards'].append(reward)
    
    # 转换为numpy数组或张量
    for key in processed_data:
        if key in ['states', 'next_states']:
            # 这些已经是张量列表，需要先堆叠成张量
            processed_data[key] = torch.stack(processed_data[key])
        elif key == 'actions':
            # 动作已经处理为适当形状，直接拼接
            processed_data[key] = torch.cat(processed_data[key], dim=0)
            
            # 如果需要将动作也扩展为多个增强版本，可以在此处理
            # 注意：需要确保states已经处理完成
            if 'states' in processed_data and isinstance(processed_data['states'], torch.Tensor):
                n_aug = processed_data['states'].shape[1]  # 获取增强数量
                processed_data[key] = processed_data[key].unsqueeze(1).expand(-1, n_aug, -1)
        else:
            processed_data[key] = torch.tensor(processed_data[key], 
                                            dtype=torch.float32 if key != 'is_expert' else torch.bool,
                                            device=device)
            if key not in ['is_expert']:
                processed_data[key] = processed_data[key].unsqueeze(1)
    
    return processed_data


def random_shift(x, pad=4):
    """
    对输入图像张量进行随机平移数据增强。
    x: [B, C, H, W]
    """
    n, c, h, w = x.shape
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    top = torch.randint(0, 2*pad, (n, 1, 1, 1), device=x.device)
    left = torch.randint(0, 2*pad, (n, 1, 1, 1), device=x.device)
    return x_pad[..., top: top+h, left: left+w]


def process_expert_batch(expert_frames, expert_actions, cave_detector, device='cuda', k_augmentations=2):
    """
    为IQL训练准备一个批次的专家数据
    
    Args:
        expert_frames: 专家帧列表
        expert_actions: 专家动作列表
        cave_detector: 洞穴检测模型
        device: 计算设备
        k_augmentations: DrQ增强次数
        
    Returns:
        批次数据元组 (obs, next_obs, actions, rewards, dones, is_expert, cave_probs, next_cave_probs)
    """
    # 预处理数据
    processed_data = preprocess_expert_data(
        expert_frames, expert_actions, 
        device=device, cave_detector=cave_detector
    )
    
    # 构建批次数据
    return (
        processed_data['states'],
        processed_data['next_states'],
        processed_data['actions'],
        processed_data['rewards'],
        processed_data['dones'],
        processed_data['is_expert'],
        processed_data['cave_probs'],
        processed_data['next_cave_probs']
    )
# 相机转换常量
MINEREC_ORIGINAL_HEIGHT_PX = 720
CAMERA_SCALER = 360.0 / 2400.0
class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(FrameSkipWrapper, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
import torch
import numpy as np
import random
from collections import deque
import torchvision.transforms as T
import torch.nn.functional as F

class ExpertPool:
    """
    专家数据池，维持固定比例的洞穴数据和探索数据
    """
    def __init__(self, max_capacity=10000, cave_ratio=0.3, device='cuda'):
        """
        初始化专家数据池
        
        参数:
            max_capacity: 池的最大容量
            cave_ratio: 需要维持的洞穴数据比例(cave_prob > 0.8)
            device: 计算设备
        """
        self.device = device
        self.max_capacity = max_capacity
        self.cave_ratio = cave_ratio
        
        # 分别存储洞穴数据和非洞穴数据
        self.cave_data = []  # cave_prob > 0.8的数据
        self.explore_data = []  # cave_prob <= 0.8的数据
        
        # 用于预处理的转换
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor()
        ])
        
        # 数据增强转换
        self.aug_transforms = [
            T.Compose([
                T.ToPILImage(),
                T.Resize((128, 128)),
                T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.05),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
                T.ToTensor()
            ]),
            T.Compose([
                T.ToPILImage(),
                T.Resize((128, 128)),
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.3)], p=0.4),
                T.ToTensor()
            ])
        ]
    
    def add(self, obs, next_obs, action, reward, done, cave_prob, next_cave_prob):
        """添加新数据到池中，维持洞穴比例"""
        # 创建数据元组
        data = (obs, next_obs, action, reward, done, cave_prob, True, next_cave_prob)
        
        # 根据洞穴概率决定添加到哪个列表
        if cave_prob > 0.8 and next_cave_prob >0.8:
            self.cave_data.append(data)
        else:
            self.explore_data.append(data)
        
        # 检查是否超过最大容量
        total_size = len(self.cave_data) + len(self.explore_data)
        if total_size > self.max_capacity:
            self._balance_pool()
    
    def add_batch(self, frames, next_frames, actions, cave_probs, next_cave_probs):
        """批量添加数据到池中"""
        for i in range(len(frames)):
            self.add(
                frames[i], 
                next_frames[i], 
                actions[i], 
                0.0,  # 默认奖励为0
                False,  # 默认非终止状态
                cave_probs[i], 
                next_cave_probs[i]
            )
    
    def _balance_pool(self):
        """平衡池内数据，保持洞穴比例"""
        total_size = len(self.cave_data) + len(self.explore_data)
        target_cave_size = int(self.max_capacity * self.cave_ratio)
        
        # 如果洞穴数据不足，不删除洞穴数据
        current_cave_size = len(self.cave_data)
        if current_cave_size < target_cave_size:
            # 只删除探索数据
            excess = total_size - self.max_capacity
            for _ in range(excess):
                if len(self.explore_data) > 0:
                    self.explore_data.pop(random.randrange(len(self.explore_data)))
        else:
            # 按比例删除两种数据
            target_explore_size = self.max_capacity - target_cave_size
            explore_excess = len(self.explore_data) - target_explore_size
            cave_excess = len(self.cave_data) - target_cave_size
            
            # 删除多余的探索数据
            for _ in range(max(0, explore_excess)):
                if len(self.explore_data) > 0:
                    self.explore_data.pop(random.randrange(len(self.explore_data)))
            
            # 删除多余的洞穴数据
            for _ in range(max(0, cave_excess)):
                if len(self.cave_data) > 0:
                    self.cave_data.pop(random.randrange(len(self.cave_data)))
    
    def sample(self, batch_size):
        """采样一批数据，保持洞穴比例"""
        # 计算要采样的洞穴数据和探索数据数量
        cave_samples = int(batch_size * self.cave_ratio)
        explore_samples = batch_size - cave_samples
        
        # 确保不超过可用数据量
        cave_samples = min(cave_samples, len(self.cave_data))
        explore_samples = min(explore_samples, len(self.explore_data))
        
        # 如果一类数据不足，从另一类多采样
        if cave_samples < int(batch_size * self.cave_ratio):
            explore_samples = min(batch_size - cave_samples, len(self.explore_data))
        if explore_samples < batch_size - int(batch_size * self.cave_ratio):
            cave_samples = min(batch_size - explore_samples, len(self.cave_data))
        
        # 随机采样
        cave_batch = random.sample(self.cave_data, cave_samples) if cave_samples > 0 else []
        explore_batch = random.sample(self.explore_data, explore_samples) if explore_samples > 0 else []
        
        # 合并批次
        batch = cave_batch + explore_batch
        random.shuffle(batch)
        
        # 解包批次数据
        obs, next_obs, actions, rewards, dones, cave_probs, is_expert, next_cave_probs = zip(*batch)
        
        # 处理观测数据
        processed_obs = []
        processed_next_obs = []
        
        for o, no in zip(obs, next_obs):
            # 应用数据增强
            aug_idx = random.randint(0, len(self.aug_transforms))
            if aug_idx < len(self.aug_transforms):
                processed_o = self.aug_transforms[aug_idx](o).unsqueeze(0)
                processed_no = self.aug_transforms[aug_idx](no).unsqueeze(0)
            else:
                processed_o = self.transform(o).unsqueeze(0)
                processed_no = self.transform(no).unsqueeze(0)
            
            # 应用随机平移
            processed_o = self._random_shift(processed_o)
            processed_no = self._random_shift(processed_no)
            
            processed_obs.append(processed_o)
            processed_next_obs.append(processed_no)
        
        # 堆叠所有处理后的张量
        obs_batch = torch.cat(processed_obs, dim=0).to(self.device)
        next_obs_batch = torch.cat(processed_next_obs, dim=0).to(self.device)
        
        # 处理动作
        processed_actions = []
        for action in actions:
            if isinstance(action, torch.Tensor):
                processed_actions.append(action.unsqueeze(0))
            else:
                # 假设有一个统一动作转换函数
                from Train import unified_convert_action
                processed_actions.append(unified_convert_action(action, self.device))
        
        actions_batch = torch.cat(processed_actions, dim=0)
        
        # 转换其他数据为张量
        rewards_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        cave_probs_batch = torch.tensor(cave_probs, dtype=torch.float32).unsqueeze(1).to(self.device)
        is_expert_batch = torch.tensor(is_expert, dtype=torch.bool).unsqueeze(1).to(self.device)
        next_cave_probs_batch = torch.tensor(next_cave_probs, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        return (
            obs_batch, next_obs_batch, actions_batch, rewards_batch, 
            dones_batch, cave_probs_batch, is_expert_batch, next_cave_probs_batch
        )
    
    def _random_shift(self, x, pad=4):
        """对输入图像张量进行随机平移数据增强"""
        n, c, h, w = x.shape
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        top = torch.randint(0, 2*pad, (n, 1, 1, 1), device=x.device)
        left = torch.randint(0, 2*pad, (n, 1, 1, 1), device=x.device)
        return x_pad[..., top: top+h, left: left+w]
    
    def __len__(self):
        """返回池中数据总量"""
        return len(self.cave_data) + len(self.explore_data)
    
    def get_statistics(self):
        """返回池中数据统计信息"""
        total = len(self)
        cave_count = len(self.cave_data)
        explore_count = len(self.explore_data)
        cave_ratio = cave_count / total if total > 0 else 0
        
        return {
            "total_samples": total,
            "cave_samples": cave_count,
            "explore_samples": explore_count,
            "current_cave_ratio": cave_ratio,
            "target_cave_ratio": self.cave_ratio
        }
def unified_convert_action(action, device):
    """
    转换专家动作为one-hot格式，返回尺寸为 [1,11] 的张量，动作索引定义如下：
      0: Move forward
      1: Move back
      2: Move left
      3: Move right
      4: Look up
      5: Look down
      6: Look left
      7: Look right
      8: Forward Jump
      9: Jump
      10: Attack
    优先级：Attack > Forward Jump > Jump > Move > Look
    """
    if isinstance(action, torch.Tensor):
        return action

    if isinstance(action, list):
        converted = [unified_convert_action(a, device) for a in action]
        return torch.cat(converted, dim=0)

    if isinstance(action, dict):
        one_hot = torch.zeros(11, dtype=torch.float32, device=device)
        # 根据优先级选择唯一动作
        camera = action.get('camera', [0, 0])
        if action.get('forward', 0) and action.get('jump', 0):
            one_hot[8] = 1.0
        elif action.get('jump', 0):
            one_hot[9] = 1.0
        elif action.get('forward', 0):
            one_hot[0] = 1.0
        elif action.get('back', 0):
            one_hot[1] = 1.0
        elif action.get('left', 0):
            one_hot[2] = 1.0
        elif action.get('right', 0):
            one_hot[3] = 1.0
        elif camera[1] <-5 :
            one_hot[6] = 1
        elif camera[1] >5 :
            one_hot[7] = 1
        elif camera[0] > 5:
            one_hot[4] = 1
        elif camera[0] < -5:
            one_hot[5] = 1
        elif  action.get('attack', 0):
            one_hot[10] = 1.0 
        else :one_hot[0] = 1.0
        return one_hot.unsqueeze(0)

    print("无法处理的动作类型，返回默认动作")
    return torch.zeros(11, dtype=torch.float32, device=device).unsqueeze(0)
class CaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),     # -> [32,35,63]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),    # -> [64,16,30]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),    # -> [64,14,28]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),      # 自适应池化到固定大小
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)
        return self.classifier(features)
    
preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((128,128)),
        T.ToTensor()
    ])
trans_pipeline = T.Compose([
        T.ToPILImage(),
        T.Resize((72,128)),
        T.ToTensor()
    ])
cave = CaveDiscriminator()
if os.path.exists('cave_detector_best.pth'):
        temp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cave_state = torch.load('cave_detector_best.pth', map_location=temp_device)['model_state_dict']
        cave.load_state_dict(cave_state)
cave = cave.to('cuda') 
cave.eval()
def get_episode_loader(dataset_dir, episode_idx=0,plot_rewards=True):
    """创建单个episode的数据加载器"""
    dataset = SimpleExpertDataset(dataset_dir)
    reward_dir = "rewards"
    os.makedirs(reward_dir, exist_ok=True)
    
    if episode_idx >= len(dataset):
        return None
    
    frames, actions = dataset.load_episode(episode_idx)
    if plot_rewards:
        try:
            # 计算该episode的rewards和cave_probs
            processed_data = process_expert_data(frames, actions, 'cuda' if torch.cuda.is_available() else 'cpu')
            rewards = processed_data['rewards']
            cave_probs = processed_data['cave_probs']
            
            # 创建新的图形
            plt.figure(figsize=(10, 5))
            plt.plot(rewards, label=f'Rewards for Episode {episode_idx}', color='blue')
            plt.plot(cave_probs, label=f'Cave Probabilities for Episode {episode_idx}', color='red')
            plt.title(f'Rewards and Cave Probabilities for Episode {episode_idx}')
            plt.xlabel('Steps')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            
            # 保存图像
            save_path = os.path.join(reward_dir, f'rewards_and_cave_probs_episode_{episode_idx}.png')
            plt.savefig(save_path)
            plt.close('all')  # 确保关闭所有图形
            
            # 计算统计信息
            avg_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            min_reward = np.min(rewards)
            avg_cave_prob = np.mean(cave_probs)
            max_cave_prob = np.max(cave_probs)
            min_cave_prob = np.min(cave_probs)
            
            print(f"\nEpisode {episode_idx} Information:")
            print(f"Average reward: {avg_reward:.4f}")
            print(f"Max reward: {max_reward:.4f}")
            print(f"Min reward: {min_reward:.4f}")
            print(f"Average cave probability: {avg_cave_prob:.4f}")
            print(f"Max cave probability: {max_cave_prob:.4f}")
            print(f"Min cave probability: {min_cave_prob:.4f}")
            print(f"Total Steps: {len(rewards)}")
            print(f"Reward and Cave Probability Curve saved to: {save_path}\n")
        
        except Exception as e:
            print(f"绘图过程中出现错误: {str(e)}")
            plt.close('all')  # 确保清理所有图形
    
    return frames, actions, episode_idx
    
def overlay_heatmap(frame, heatmap, alpha=0.4):
    """
    将 heatmap 叠加到 frame 上
    Args:
        frame: 原始 RGB 图像，numpy 数组，形状 [H,W,3], dtype uint8
        heatmap: 热力图，numpy 数组，范围 [0,1]，形状应与 frame 前两维相同
        alpha: 透明度权重
    Returns:
        叠加后的 BGR 图像，用于视频保存
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, frame, 1 - alpha, 0)
    # 转换为 BGR（因为OpenCV保存视频要求 BGR 格式）
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    return overlay_bgr
def random_shift(x, pad=4):
    """
    对输入图像张量进行随机平移数据增强。
    x: [B, C, H, W]
    """
    n, c, h, w = x.shape
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    top = torch.randint(0, 2*pad, (n, 1, 1, 1), device=x.device)
    left = torch.randint(0, 2*pad, (n, 1, 1, 1), device=x.device)
    return x_pad[..., top: top+h, left: left+w]


class OnlineMemory:
    def __init__(self, capacity=30000):
        self.memory = []
        self.capacity = capacity
    def store(self, obs, next_obs, action, reward, done, cave_prob,next_cave_prob):
        self.memory.append((obs, next_obs, action, reward, done, cave_prob, False,next_cave_prob))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
    def sample(self, batch_size):
        
        indices = np.random.choice(len(self.memory), batch_size)
        samples = [self.memory[i] for i in indices]
        obs, next_obs, actions, rewards, dones, cave_probs, is_expert ,next_cave_prob= zip(*samples)
        # 处理观测数据
        processed_obs = []
        processed_next_obs = []
        
        for o, no in zip(obs, next_obs):
            # 分别处理每个观测
            processed_o = preprocess(o).unsqueeze(0)  
            processed_no = preprocess(no).unsqueeze(0)
            processed_obs.append(processed_o)
            processed_next_obs.append(processed_no)
            
        # 堆叠所有处理后的张量
        obs_batch = torch.cat(processed_obs, dim=0).to('cuda')
        next_obs_batch = torch.cat(processed_next_obs, dim=0).to('cuda')
        
        return (obs_batch,
                next_obs_batch,
                list(actions),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to('cuda'),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to('cuda'),
                torch.tensor(cave_probs, dtype=torch.float32).unsqueeze(1).to('cuda'),
                torch.tensor(is_expert, dtype=torch.bool).unsqueeze(1).to('cuda'),
                torch.tensor(next_cave_prob, dtype=torch.float32).unsqueeze(1).to('cuda'))
def clean_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
def save_video(frames, path, agent=None, target_action="forward", fps=20, cave_probs=None, use_gradcam=True):
    """
    保存视频帧为 MP4 文件，可在帧上叠加 Grad-CAM 热力图
    Args:
        frames: 包含原始 RGB 帧的列表
        path: 输出视频路径
        agent: 用于计算 Grad-CAM 的 IQLearnAgent 实例
        target_action: 指定计算热力图时的目标离散动作名称（例如 "forward"）
        fps: 帧率
        cave_probs: 可选的洞穴概率列表（原有功能）
        use_gradcam: 是否计算并叠加 Grad-CAM 热力图
    """
    if agent != None :
            
        if len(frames) == 0:
            return
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
        
        # 确保 target_action 在 ACTION_KEYS[:-1] 范围内
        discrete_keys = agent.__class__.__dict__.get('ACTION_KEYS', ACTION_KEYS[:-1])
        # 此处默认 target_action 的索引,例如 "forward" 对应索引 2
        try:
            target_index = ACTION_KEYS[:-1].index(target_action)
        except ValueError:
            target_index = 0
        
        for idx, frame in enumerate(frames):
            # frame为 RGB，dtype uint8
            overlay_img = frame.copy()
            if use_gradcam:
                # 将 frame 转换为 tensor，并使用 agent.transform（或 T.ToTensor()）处理
                image_tensor = agent.transform(frame).to(agent.device)  # shape [3,H,W]
                image_tensor = image_tensor.unsqueeze(0)
                # 计算 Grad-CAM 热力图
                agent.zero_grad()
                cam = agent.compute_gradcam(image_tensor, target_index)  # [1,1,H,W]
                heatmap = cam.squeeze().detach().cpu().numpy()  # shape [H,W]
                # 将热力图 resize 到 frame 尺寸（若尺寸不同）
                if heatmap.shape[:2] != (height, width):
                    heatmap = cv2.resize(heatmap, (width, height))
                overlay_img = overlay_heatmap(frame, heatmap, alpha=0.4)
            # 如果提供 cave_probs，也可在图片上绘制文字（保留原有功能）
            if cave_probs is not None and idx < len(cave_probs):
                text = f"cave_prob: {cave_probs[idx]:.2f}"
                cv2.putText(overlay_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(overlay_img)
        out.release()
    else:
        if len(frames) == 0:
            return
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
        for idx, frame in enumerate(frames):
            # frame为 RGB，dtype uint8
            overlay_img = frame.copy()
            # 如果提供 cave_probs，也可在图片上绘制文字
            if cave_probs is not None and idx < len(cave_probs):
                text = f"cave_prob: {cave_probs[idx]:.2f}"
                cv2.putText(overlay_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
            # 添加此行，转换RGB到BGR
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            out.write(overlay_img)
        out.release()

def main(args):
    
    import torch
    torch.autograd.set_detect_anomaly(True)
    os.makedirs(args.video_dir, exist_ok=True)
    swanlab.init(project="no-VPT-Train", entity="swanlab")
    device = torch.device(args.device)
    env = gym.make('MineRLBasaltFindCave-v0')
    env = FrameSkipWrapper(env, skip=4)
    print("---Loading model---")
    # 初始化专家数据加载器（假设专家数据存放在目录 args.expert_dir）
    expert_loader = DT(dataset_dir=args.expert_dir, n_workers=args.n_workers, batch_size=args.expert_batch_size)
    #expert_loader = ContinuousExpertLoader(dataset_dir=args.expert_dir, batch_size=args.expert_batch_size)
    iql_agent = IQLearnAgent(args=args).to(device)
    if os.path.exists(args.iql_weights):
        iql_agent.load_state_dict(torch.load(args.iql_weights, map_location=device))
        print(f"已加载预训练 IQL 权重: {args.iql_weights}")
    else:
        print("没有找到IQL权重")
    actor_optimizer = optim.Adam([
        {'params': iql_agent.discrete.parameters()},

    ], lr=3e-4)
    
    critic_optimizer = optim.Adam([
    {'params': iql_agent.critic_sa.parameters()},

], lr=3e-4)
    
    # 分别创建学习率调度器
    actor_scheduler = CyclicLR(
        actor_optimizer,
        base_lr=3e-4,
        max_lr=1e-3,
        step_size_up=500,
        mode='triangular2'
    )
    
    critic_scheduler = CyclicLR(
        critic_optimizer,
        base_lr=3e-4,
        max_lr=1e-3,
        step_size_up=500,
        mode='triangular2'
    )
    online_memory = OnlineMemory(capacity=args.online_capacity)
    #expert_loader = DataLoader(dataset_dir=args.expert_dir, n_workers=args.n_workers, batch_size=args.expert_batch_size)
    expert_pool = ExpertPool(max_capacity=args.expert_pool_size, cave_ratio=0.3)
    num_updates = args.num_updates
    
    expert_iter = iter(expert_loader)


    for _ in range(5000):
            try:
                expert_frames, expert_actions, _ = next(expert_iter)
                    # 获取洞穴概率
                cave_probs = []
                next_cave_probs = []
                for i in range(len(expert_frames)-1):  # 注意这里要减1，因为我们需要next_frame
                        with torch.no_grad():
                            curr_frame = trans_pipeline(expert_frames[i]).unsqueeze(0).to(device)
                            next_frame = trans_pipeline(expert_frames[i+1]).unsqueeze(0).to(device)
                            
                            curr_logits = cave(curr_frame)
                            next_logits = cave(next_frame)
                            
                            curr_probs = F.softmax(curr_logits, dim=1)
                            next_probs = F.softmax(next_logits, dim=1)
                            
                            cave_probs.append(curr_probs[0, 1].item())
                            next_cave_probs.append(next_probs[0, 1].item())
                    
                    # 添加到专家池
                expert_pool.add_batch(
                        expert_frames[:-1],  # 除了最后一帧
                        expert_frames[1:],   # 除了第一帧
                        expert_actions[:-1], # 对应的动作
                        cave_probs,
                        next_cave_probs
                    )
                
                # 更新进度条
              
                
                # 每1000个样本显示一次池统计信息
                if len(expert_pool) % 1000 < 20:
                    stats = expert_pool.get_statistics()
                    print(f"\n专家池: {stats['total_samples']}/{args.expert_pool_size}, "
                        f"洞穴比例: {stats['current_cave_ratio']:.2%}")
                    
            except StopIteration:
                expert_iter = iter(expert_loader)
        
      
             
    print("成功加载矿洞判别模型")
    print("开始训练 IQ-Learn 模型...")

    
    for update in range(num_updates):
        if update == 0 :
            continue
        # 获取专家数据并填充专家池
        if update % args.expert_fill_interval == 0:
            print("直接填充十组专家数据")
            for _ in range(10):
                try:
                        expert_frames, expert_actions, _ = next(expert_iter)
                        # 获取洞穴概率
                        cave_probs = []
                        next_cave_probs = []
                        for i in range(len(expert_frames)-1):  # 注意这里要减1，因为我们需要next_frame
                            with torch.no_grad():
                                curr_frame = trans_pipeline(expert_frames[i]).unsqueeze(0).to(device)
                                next_frame = trans_pipeline(expert_frames[i+1]).unsqueeze(0).to(device)
                                
                                curr_logits = cave(curr_frame)
                                next_logits = cave(next_frame)
                                
                                curr_probs = F.softmax(curr_logits, dim=1)
                                next_probs = F.softmax(next_logits, dim=1)
                                
                                cave_probs.append(curr_probs[0, 1].item())
                                next_cave_probs.append(next_probs[0, 1].item())
                        
                        # 添加到专家池
                        expert_pool.add_batch(
                            expert_frames[:-1],  # 除了最后一帧
                            expert_frames[1:],   # 除了第一帧
                            expert_actions[:-1], # 对应的动作
                            cave_probs,
                            next_cave_probs
                        )
                        
                        # 打印专家池统计信息
                        if update % 100 == 0:
                            stats = expert_pool.get_statistics()
                            print(f"专家池统计: 总样本={stats['total_samples']}, "
                                f"洞穴样本={stats['cave_samples']} ({stats['current_cave_ratio']:.2%}), "
                                f"探索样本={stats['explore_samples']}")
                except StopIteration:
                        expert_iter = iter(expert_loader)
        swanlab.log({"update": update})
        
        iql_agent.reset_memory(batch_size=1)
        #######################
        is_online_phase = 1 if update >=0 else 0
        ######################
        if is_online_phase:
            clean_memory()
            try:
                # 尝试获取新的专家数据
                expert_frames, expert_actions, _ = next(expert_iter)
                #print("专家原动作：",expert_actions,'\n')
                #一次获取8帧专家数据
                expert_actions=unified_convert_action(expert_actions, device)
                #print("处理后的专家动作：",expert_actions)
            except StopIteration:
                # 如果迭代器结束，重新创建迭代器
                print("重置专家数据迭代器")
                expert_iter = iter(expert_loader)
                expert_frames, expert_actions, _ = next(expert_iter)
            if update %50 == 0 or update ==1 :
                frames_buffer = []  # 用于存储当前episode的帧
                cave_prob_buffer=[]
                done = False
                env = gym.make('MineRLBasaltFindCave-v0')
                obs = env.reset()
                episode_reward = 0
                iql_agent.reset_memory(batch_size=1)  # 添加这行
                step=0
                pov_tensor = trans_pipeline(obs['pov']).unsqueeze(0).to('cuda')
                with torch.no_grad():
                        logits = cave(pov_tensor)
                        probs = F.softmax(logits, dim=1)  # 添加softmax
                        cave_prob = probs[:, 1].unsqueeze(1)
                        cave_prob_value = cave_prob.detach().cpu().item()
                
                while not done:
                    step+=1
                    current_frame = env.render(mode='rgb_array')  # 获取当前帧
                    frames_buffer.append(current_frame)  # 保存帧
                    
                    
                    cave_prob_buffer.append(cave_prob_value)
                    # 移除 batch 维度，将图像移回 CPU，以便 agent.transform 正常工作
                    
                    action = iql_agent.get_action(obs,cave_prob)
                    #print("agent给出的动作：",action)
                    if 'ESC' not in action:
                        action['ESC'] = 0
                    #print(action)
                    
                    next_obs, reward, done, _ = env.step(action)
                    next_pov_tensor=trans_pipeline(next_obs['pov']).unsqueeze(0).to('cuda')
                    with torch.no_grad():
                        logits = cave(next_pov_tensor)
                        probs = F.softmax(logits, dim=1)  # 添加softmax
                        next_cave_prob = probs[:, 1].unsqueeze(1)
                        next_cave_prob = next_cave_prob.detach().cpu().item()
                    
                    #shaped_reward =  10 * cave_prob if cave_prob >0.9 else 0
                    shaped_reward=0
                    online_memory.store(obs['pov'], next_obs['pov'], action, shaped_reward, done, cave_prob_value,next_cave_prob)
                    obs = next_obs
                    cave_prob_value = next_cave_prob
                    episode_reward += reward
                    if done or step >= 1000:
                        # 保存本次episode的视频
                        if len(frames_buffer) > 0:
                            video_path = os.path.join(
                                args.video_dir, 
                                f"online_exploration_update_{update}_step_{step}.mp4"
                            )
                            if update % 20 ==0 :
                                save_video(frames_buffer, video_path,agent = None, fps=20, cave_probs=cave_prob_buffer) 
                            else :
                                save_video(frames_buffer, video_path,agent = None, fps=20, cave_probs=cave_prob_buffer)
                            print(f"在线探索视频已保存至: {video_path}")
                        frames_buffer = []  # 重置缓冲区
                        env.close()
                        #del obs, next_obs, actions_batch, rewards_batch, dones_batch, cave_probs_batch
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    if step >= 1000 :
                        done = True
                    # 混合训练：同时使用在线数据和专家数据
         
            if len(expert_pool) > args.batch_size and len(online_memory.memory) > args.batch_size:
                    # 重置LSTM状态 - 每个批次都重置，避免错误的时序关联
                iql_agent.reset_memory(batch_size=args.batch_size * 2)  # 总批次大小=专家+在线
                
                # 从两个来源采样数据
                expert_batch = expert_pool.sample(args.batch_size)
                online_batch = online_memory.sample(args.batch_size)
                
                # 解包专家数据 - 已经处理好的张量格式
                obs_expert, next_obs_expert, actions_expert, rewards_expert, \
                dones_expert, cave_probs_expert, is_expert_expert, next_cave_probs_expert = expert_batch
                
                # 解包在线数据 - 已经处理好的张量格式
                obs_online, next_obs_online, actions_online_raw, rewards_online, \
                dones_online, cave_probs_online, is_expert_online, next_cave_probs_online = online_batch
                
                # 统一处理在线动作格式 - 确保与专家动作格式一致
                processed_online_actions = []
                for act in actions_online_raw:
                    converted = unified_convert_action(act, device)
                    processed_online_actions.append(converted)
                
                # 如果是单个动作，可能需要额外的挤压操作
                if len(processed_online_actions) == 1:
                    actions_online = processed_online_actions[0]
                else:
                    actions_online = torch.cat(processed_online_actions, dim=0)
                
                # 规范化维度，确保所有批次数据具有一致的形状
                if cave_probs_expert.dim() == 1:
                    cave_probs_expert = cave_probs_expert.unsqueeze(1)
                if next_cave_probs_expert.dim() == 1:
                    next_cave_probs_expert = next_cave_probs_expert.unsqueeze(1)
                if cave_probs_online.dim() == 1:
                    cave_probs_online = cave_probs_online.unsqueeze(1)
                if next_cave_probs_online.dim() == 1:
                    next_cave_probs_online = next_cave_probs_online.unsqueeze(1)
                
                # 统一专家标志维度
                if is_expert_expert.dim() != is_expert_online.dim():
                    if is_expert_expert.dim() > is_expert_online.dim():
                        is_expert_online = is_expert_online.unsqueeze(1)
                    else:
                        is_expert_expert = is_expert_expert.unsqueeze(1)
                    
                # 日志记录
                if update % 50 == 0:
                    print(f"批次信息: 专家={obs_expert.shape}, 在线={obs_online.shape}")
                    print(f"洞穴比例: 专家={cave_probs_expert.mean().item():.3f}, 在线={cave_probs_online.mean().item():.3f}")
                
                # 合并专家和在线数据
                obs_batch = torch.cat([obs_expert, obs_online], dim=0)
                next_obs_batch = torch.cat([next_obs_expert, next_obs_online], dim=0)
                rewards_batch = torch.cat([rewards_expert, rewards_online], dim=0)
                dones_batch = torch.cat([dones_expert, dones_online], dim=0)
                cave_probs_batch = torch.cat([cave_probs_expert, cave_probs_online], dim=0)
                next_cave_probs_batch = torch.cat([next_cave_probs_expert, next_cave_probs_online], dim=0)
                is_expert_batch = torch.cat([is_expert_expert, is_expert_online], dim=0)
                
                # 合并动作 - 确保是2D格式 [batch_size, action_dim]
                if actions_expert.dim() > 2:
                    actions_expert = actions_expert.view(-1, actions_expert.size(-1))
                if actions_online.dim() > 2:
                    actions_online = actions_online.view(-1, actions_online.size(-1))
                    
                actions_batch = torch.cat([actions_expert, actions_online], dim=0)
                
                # 计算重要性权重 - 可选，用于平衡数据分布
                weights = torch.ones_like(rewards_batch)
                # 增强对发现洞穴的权重
                discovering_cave = (cave_probs_batch < 0.3) & (next_cave_probs_batch > 0.5)
                if discovering_cave.any():
                    weights[discovering_cave] *= 3.0
                    
                # 组装批次数据，包含重要性权重
                batch_data = (obs_batch, next_obs_batch, actions_batch, rewards_batch, 
                            dones_batch, is_expert_batch, cave_probs_batch, 
                            next_cave_probs_batch, weights)
        
                            
            
                critic_loss, actor_loss = iql_agent.compute_iql_loss(batch_data)
                # 然后更新actor
                swanlab.log({"critic_loss":critic_loss.item()})
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(iql_agent.critic_sa.parameters(), 1.0)
                critic_optimizer.step()

                critic_loss, actor_loss = iql_agent.compute_iql_loss(batch_data)
                actor_optimizer.zero_grad()

                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(iql_agent.discrete.parameters(), 1.0)
                actor_optimizer.step()
                
                
                swanlab.log({"actor_loss":actor_loss.item()})
                        # 更新学习率
                actor_scheduler.step()
                critic_scheduler.step()
                # 清理内存
                del expert_batch, online_batch
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                                
                   
                iql_agent.update_alpha()
                    
                if update % args.target_update_interval == 0:  # 例如每100步更新一次
                    iql_agent.soft_update(iql_agent.critic_sa, iql_agent.critic_target)
        
                
                # 建议在每个大循环结束后清理显存
            
                if update % args.eval_interval == 0:
                    
                    save_path = os.path.join(args.save_dir, "iql_agent.pth")
                    torch.save(iql_agent.state_dict(), save_path)
                    print(f"新模型已保存")

                torch.cuda.empty_cache() 
                print("训练结束。")
                
        else:
            # 如果数据不足，继续收集
            print(f"数据不足，跳过更新。专家池: {len(expert_pool)}/{args.batch_size}，在线记忆: {len(online_memory.memory)}/{args.batch_size}")
            continue
                    
        

        
if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(
        device="cuda",
        gamma=0.99,
        expert_loss_coef=0.1,
        expert_dir="F:/MineRL/VPT/data",
        online_capacity=10000,
        expert_pool_size=10000,
        batch_size=8,# 在线训练的批量大小
        num_updates=72000,
        warmup_steps=10000,
        online_steps=1000,
        log_interval=10,
        save_dir="F:/MineRL/VPT/model_save",
        iql_weights="F:/MineRL/VPT/model_save/iql_agent.pth",
        video_interval=10,
        video_dir="F:/MineRL/VPT/videos",
        bc_loss_coef=0.01,
        critic_tau=0.005,
        eval_interval=5,  # 评估间隔
        online_start_update=10,  # 开始在线训练的更新步数
        init_alpha=0.1,           # 初始温度值
        target_entropy=-8,        # 目标熵值
        alpha_lr=3e-4,           # alpha学习率
        min_alpha=0.01,          # 最小温度值
        max_alpha=1.0,           # 最大温度值
        clean_memory_interval=10,  # 清理显存间隔
        n_workers=9,  # 数据加载器的工作线程数
        expert_batch_size=9,  # 专家
        target_update_interval=10,  # 
        expert_fill_interval=5,
        #num_updates=1000000,  # 总更新次数
    )
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)