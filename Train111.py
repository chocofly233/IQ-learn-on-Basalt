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
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from IQLAgent import IQLearnAgent     # 独立训练的 IQ-Learn 模型
from agent import MineRLAgent, ENV_KWARGS   # 预训练的 VPT 模型
from Video_data_loader import DataLoader    # 专家数据加载器
from path_tracker import PathTracker
import json
import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
RESOLUTION = (72,128)
ACTION_KEYS = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak',
               'sprint', 'use', 'drop', 'inventory', 'hotbar.1', 'hotbar.2',
               'hotbar.3', 'hotbar.4', 'hotbar.5', 'hotbar.6', 'hotbar.7',
               'hotbar.8', 'hotbar.9', 'camera']

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
class SimpleExpertDataset(Dataset):
    def __init__(self, dataset_dir, resolution=(128, 128)):
        self.dataset_dir = dataset_dir
        self.resolution = resolution
        
        # 获取所有视频文件路径
        self.video_files = sorted(glob.glob(os.path.join(dataset_dir, "*.mp4")))
        self.json_files = [f.replace('.mp4', '.jsonl') for f in self.video_files]
        
        # 过滤掉不存在对应json文件的视频
        valid_pairs = []
        for video_f, json_f in zip(self.video_files, self.json_files):
            if os.path.exists(json_f):
                valid_pairs.append((video_f, json_f))
        
        self.data_pairs = valid_pairs
        print(f"找到 {len(self.data_pairs)} 个有效的视频")

    def __len__(self):
        return len(self.data_pairs)

    def load_episode(self, index):
        """加载单个视频的所有帧和动作"""
        video_path, json_path = self.data_pairs[index]
        
        # 读取JSON数据
        with open(json_path) as f:
            json_lines = f.readlines()
            actions = [json.loads(line) for line in json_lines]
        
        # 读取视频帧
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为RGB并调整大小
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resolution)
            frames.append(frame)
        cap.release()
        
        print(f"从视频 {os.path.basename(video_path)} 加载了 {len(frames)} 帧")
        return frames, actions

def process_expert_data(frames, actions, device):
    processed_data = {
        'obs': [], 'next_obs': [], 'actions': [], 
        'rewards': [], 'dones': [], 'cave_probs': []
    }
    
    n_steps = min(len(frames)-1, len(actions)-1)
    
    for i in range(n_steps):
        frame = frames[i]
        # 转换为张量并调整维度
        curr_frame = preprocess(frame).to(device)
        next_frame = preprocess(frames[i+1]).to(device)
        
        # 计算洞穴概率
        cave_frame = trans_pipeline(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = cave(cave_frame)
            # 添加softmax处理
            probs = F.softmax(logits, dim=1)
            cave_prob = probs[0, 1].item()
            print(cave_prob)
            if i == 0:
                print(f"Frame shape: {frame.shape}")
                print(f"Cave frame shape: {cave_frame.shape}")
                print(f"Raw probabilities: {probs[0].cpu().numpy()}")
                print(f"Cave probability: {cave_prob:.4f}")
                
        # 使用一致的奖励计算方式
        reward = min(10, -np.log(1 - cave_prob + 1e-6))
        
        processed_data['obs'].append(curr_frame)
        processed_data['next_obs'].append(next_frame)
        processed_data['actions'].append(actions[i])
        processed_data['rewards'].append(reward)
        processed_data['cave_probs'].append(cave_prob)  # 直接存储洞穴概率
        processed_data['dones'].append(1.0 if i == n_steps-1 else 0.0)
    
    return processed_data

def get_episode_loader(dataset_dir, episode_idx=0,plot_rewards=False):
    """创建单个episode的数据加载器"""
    dataset = SimpleExpertDataset(dataset_dir)
    if episode_idx >= len(dataset):
        return None
    
    frames, actions = dataset.load_episode(episode_idx)
    if plot_rewards:
        # 计算该episode的rewards和cave_probs
        processed_data = process_expert_data(frames, actions, 'cuda' if torch.cuda.is_available() else 'cpu')
        rewards = processed_data['rewards']
        cave_probs = processed_data['cave_probs']
        
        # 绘制曲线
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label=f'Rewards for Episode {episode_idx}', color='blue')
        plt.plot(cave_probs, label=f'Cave Probabilities for Episode {episode_idx}', color='red')
        plt.title(f'Rewards and Cave Probabilities for Episode {episode_idx}')
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # 保存图像
        save_path = f'rewards_and_cave_probs_episode_{episode_idx}.png'
        plt.savefig(save_path)
        plt.close()
        
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
    frames, actions = dataset.load_episode(episode_idx)
    
    # 修改返回值，不要返回None
    return frames, actions, episode_idx
    

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

def unified_convert_action(action, device):
    """
    将动作转换为一个平坦的 Tensor。支持输入为 dict、list 或 Tensor 类型。
    对于 dict 类型动作：
      - 除 camera 外的离散动作按照 ACTION_KEYS 顺序取值（转换为 int）
      - camera 部分转换为 2 维浮点列表
    """
    if isinstance(action, torch.Tensor):
        return action
    if isinstance(action, list):
        converted = [unified_convert_action(a, device) for a in action]
        return torch.cat(converted, dim=0)
    if isinstance(action, dict):
        discrete_keys = ACTION_KEYS[:-1]
        discrete_values = []
        for key in discrete_keys:
            if key in action:
                val = action[key]
                if isinstance(val, np.ndarray):
                    if val.size == 1:
                        val = int(val.item())
                    else:
                        val = int(val.flatten()[0])
                elif hasattr(val, 'item'):
                    val = int(val.item())
                else:
                    val = int(val)
            else:
                val = 0
            discrete_values.append(val)
        camera_val = action.get('camera', None)
        if camera_val is None:
            camera_values = [0.0, 0.0]
        else:
            if isinstance(camera_val, np.ndarray):
                camera_values = camera_val.flatten().tolist()
            elif isinstance(camera_val, (list, tuple)):
                camera_values = list(camera_val)
            elif hasattr(camera_val, 'tolist'):
                camera_values = camera_val.tolist()
            else:
                camera_values = [float(camera_val)]
            if len(camera_values) < 2:
                camera_values = camera_values + [0.0]*(2 - len(camera_values))
            elif len(camera_values) > 2:
                camera_values = camera_values[:2]
        combined = discrete_values + camera_values
        return torch.tensor(combined, dtype=torch.float32, device=device).unsqueeze(0)
    return None


class OnlineMemory:
    def __init__(self, capacity=1000):
        self.memory = []
        self.capacity = capacity
    def store(self, obs, next_obs, action, reward, done, cave_prob):
        self.memory.append((obs, next_obs, action, reward, done, cave_prob, False))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
    def sample(self, batch_size):
        import random
        cave_probs = np.array([item[5] for item in self.memory]) + 1e-6
        weights = cave_probs / cave_probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=weights)
        samples = [self.memory[i] for i in indices]
        obs, next_obs, actions, rewards, dones, cave_probs, is_expert = zip(*samples)
        a = np.array(obs)
        b = torch.tensor(a)
        c = np.array(next_obs)
        d = torch.tensor(c)
        return (b,
                d,
                list(actions),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
                torch.tensor(cave_probs, dtype=torch.float32).unsqueeze(1),
                torch.tensor(is_expert, dtype=torch.bool).unsqueeze(1))
def evaluate(agent, env, num_episodes=5, video_path=None):
        """评估智能体并录制视频"""
        total_reward = 0
        all_frames = []
        
        for episode in range(num_episodes):
            done = False
            obs = env.reset()
            episode_frames = []
            episode_reward = 0
            # 每个episode开始时重置LSTM状态
            agent.reset_memory(batch_size=1)  # 添加这行
            while not done:
                # 获取动作
                pov = obs["pov"]  # 获取第一人称视角图像
                obs_tensor = trans_pipeline(pov).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    logits = cave(obs_tensor)
                    # 添加softmax处理
                    probs = F.softmax(logits, dim=1)
                    cave_prob = probs[0, 1].item()
        
                cave_prob_tensor = torch.tensor([[cave_prob]], device=agent.device)  # 添加这行
                action = agent.get_action(obs,cave_prob_tensor)
                # 执行动作
                obs, reward, done, info = env.step(action)
                reward=min(10,-np.log(1-cave_prob+1e-6))
                episode_reward += reward
                
                # 保存帧
                if video_path is not None:
                    frame = env.render(mode='rgb_array')
                    episode_frames.append(frame)
            
            total_reward += episode_reward
            if len(episode_frames) > 0:
                all_frames.extend(episode_frames)
                
        # 保存视频
        if video_path is not None and len(all_frames) > 0:
            height, width = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 15, (width, height))
            
            for frame in all_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"评估视频已保存到: {video_path}")
            
        return total_reward / num_episodes
def main(args):
    os.makedirs(args.video_dir, exist_ok=True)
    swanlab.init(project="no-VPT-Train", entity="swanlab")
    device = torch.device(args.device)
    env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")

    iql_agent = IQLearnAgent(args=args).to(device)
    if os.path.exists(args.iql_weights):
        iql_agent.load_state_dict(torch.load(args.iql_weights, map_location=device))
        print(f"已加载预训练 IQL 权重: {args.iql_weights}")
    else:
        print("没有找到IQL权重")
    optimizer = optim.Adam(iql_agent.parameters(), lr=3e-4)
    scheduler = CyclicLR(
        optimizer,
        base_lr=3e-4,
        max_lr=3e-3,
        step_size_up=2000,
        mode='triangular2'
    )
    online_memory = OnlineMemory(capacity=args.online_capacity)
    #expert_loader = DataLoader(dataset_dir=args.expert_dir, n_workers=args.n_workers, batch_size=args.expert_batch_size)
    
    total_steps = 0
    num_updates = args.num_updates
    batch_size = args.batch_size

    print("成功加载矿洞判别模型")
    print("开始训练 IQ-Learn 模型...")
    dataset = SimpleExpertDataset(args.expert_dir)
    current_episode = 0
    is_expert_data_exhausted = False

    for update in range(num_updates):
        if update ==0 :
            continue 
        video_frames = []
        iql_agent.reset_memory(batch_size=1)
        iql_agent.update_alpha(update)
        trajectory = []
        is_online_phase = is_expert_data_exhausted and update >= args.online_start_update
        if is_online_phase:
            # 在线数据收集
            done = False
            obs = env.reset()
            episode_reward = 0
            iql_agent.reset_memory(batch_size=1)  # 添加这行
            step=0
            while not done:
                step+=1
                action = iql_agent.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                
                obs_tensor = trans_pipeline(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = cave(obs_tensor)
                    # 添加softmax处理
                    probs = F.softmax(logits, dim=1)
                    cave_prob = probs[0, 1].item()
        
                shaped_reward = -np.log(1 - cave_prob + 1e-6)
                
                online_memory.store(obs, next_obs, action, shaped_reward, done, cave_prob)
                
                # 混合训练：同时使用在线数据和专家数据
                if len(online_memory.memory) > args.batch_size:
                    # 采样在线数据
                    online_batch = online_memory.sample(args.batch_size // 2)
                    
                    # 采样专家数据
                    expert_episode = random.randint(0, 2665 - 1)
                    expert_data = get_episode_loader(args.expert_dir, expert_episode)
                    if expert_data:
                        frames, actions, _ = expert_data
                        processed_expert = process_expert_data(frames, actions, device)
                        
                        # 随机选择专家数据批次
                        if len(processed_expert['obs']) > args.batch_size // 2:
                            idx = random.randint(0, len(processed_expert['obs']) - args.batch_size // 2)
                            expert_obs = torch.stack(processed_expert['obs'][idx:idx + args.batch_size // 2]).to(device)
                            expert_next_obs = torch.stack(processed_expert['next_obs'][idx:idx + args.batch_size // 2]).to(device)
                            expert_actions = unified_convert_action(processed_expert['actions'][idx:idx + args.batch_size // 2], device)
                            expert_rewards = torch.tensor(processed_expert['rewards'][idx:idx + args.batch_size // 2], device=device).unsqueeze(1)
                            expert_dones = torch.tensor(processed_expert['dones'][idx:idx + args.batch_size // 2], device=device).unsqueeze(1)
                            expert_cave_probs = torch.tensor(processed_expert['cave_probs'][idx:idx + args.batch_size // 2], device=device).unsqueeze(1)
                            
                            # 合并在线数据和专家数据
                            combined_obs = torch.cat([online_batch[0], expert_obs], dim=0)
                            combined_next_obs = torch.cat([online_batch[1], expert_next_obs], dim=0)
                            combined_actions = torch.cat([unified_convert_action(online_batch[2], device), expert_actions], dim=0)
                            combined_rewards = torch.cat([online_batch[3], expert_rewards], dim=0)
                            combined_dones = torch.cat([online_batch[4], expert_dones], dim=0)
                            combined_cave_probs = torch.cat([online_batch[5], expert_cave_probs], dim=0)
                            combined_is_expert = torch.cat([
                                torch.zeros(args.batch_size // 2, 1, dtype=torch.bool, device=device),
                                torch.ones(args.batch_size // 2, 1, dtype=torch.bool, device=device)
                            ], dim=0)
                            iql_agent.reset_memory(batch_size=args.batch_size)  # 完整批次大小
                            # 训练
                            batch_data = (combined_obs, combined_next_obs, combined_actions, 
                                        combined_rewards, combined_dones, combined_is_expert, 
                                        combined_cave_probs)
                            
                            optimizer.zero_grad()
                            loss, loss_dict = iql_agent.compute_iql_loss(batch_data)
                            loss.backward()
                            optimizer.step()
                
                obs = next_obs
                episode_reward += reward
                if step >= 1000 :
                    done=True
        else :
            # 专家数据训练阶段
            data = get_episode_loader(args.expert_dir, current_episode,plot_rewards=True)
            if data is None:
                current_episode = 0
                if not is_expert_data_exhausted:
                    is_expert_data_exhausted = True
                    print("专家数据训练完成，开始混合训练阶段")
                data = get_episode_loader(args.expert_dir, current_episode)
            if data :
                frames, actions, _ = data
                iql_agent.reset_memory(batch_size=1)  # 训练时使用指定的batch_size
                # 获取所有 worker 的数据
                processed_data = process_expert_data(frames, actions, device)
                # 批量训练
                for i in range(0, len(processed_data['obs']), args.batch_size):
                    batch_end = min(i + args.batch_size, len(processed_data['obs']))
                    
                    obs_batch = torch.stack(processed_data['obs'][i:batch_end]).to(device)
                    next_obs_batch = torch.stack(processed_data['next_obs'][i:batch_end]).to(device)
                    actions_batch = unified_convert_action(processed_data['actions'][i:batch_end], device)
                    rewards_batch = torch.tensor(processed_data['rewards'][i:batch_end], dtype=torch.float32, device=device).unsqueeze(1)
                    dones_batch = torch.tensor(processed_data['dones'][i:batch_end], dtype=torch.float32, device=device).unsqueeze(1)
                    cave_probs_batch = []
                    for f in frames[i:batch_end]:
                        f_tensor = trans_pipeline(f).unsqueeze(0).to(device)
                        prob = cave(f_tensor)[0, 1].item()
                        cave_probs_batch.append(prob)
                    cave_probs_batch = torch.tensor(cave_probs_batch, device=device).unsqueeze(1)
                    is_expert_batch = torch.ones(obs_batch.shape[0], 1, dtype=torch.bool, device=device)
                    # 训练步骤
                    optimizer.zero_grad()
                    batch_data = (obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch, is_expert_batch, cave_probs_batch)
                    loss, loss_dict = iql_agent.compute_iql_loss(batch_data)
                    swanlab.log({"total_loss": loss_dict["total_loss"],
                                "online_loss": loss_dict["online_loss"],
                                "bc_loss": loss_dict["bc_loss"],
                                "expert_loss": loss_dict["expert_loss"]})
                    loss.backward()
                    optimizer.step()
                
            
            # 每 args.video_interval 个 update 保存一次视频（使用 cv2.VideoWriter）       
        if update % args.log_interval == 0:
                print(f"Update {update}: Total Loss {loss_dict['total_loss']:.4f} "
                    f"online_loss {loss_dict['online_loss']:.4f} expert_loss {loss_dict['expert_loss']:.4f} "
                    f"bc_loss {loss_dict['bc_loss']:.4f}")
        if update % args.video_interval == 0:
                iql_agent.save(os.path.join(args.save_dir, "iql_agent.pth"))
        current_episode += 1
        if current_episode >= len(dataset):
                current_episode = 0
                scheduler.step()
        # 在训练循环中定期评估
        if update % args.eval_interval == 0:
            eval_reward = evaluate(iql_agent, env)
            swanlab.log({"eval_reward": eval_reward})
        # 建议在每个大循环结束后清理显存
        if update % args.clean_memory_interval == 0:
            torch.cuda.empty_cache()       
        iql_agent.save(os.path.join(args.save_dir, "iql_agent.pth"))
        print("训练结束，模型已保存。")
        
if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(
        device="cuda",
        gamma=0.99,
        expert_loss_coef=10.0,
        lr=1e-4,
        vpt_weights="F:/MineRL/VPT/finetuned-2x.weights",
        expert_dir="F:/MineRL/VPT/data",
        online_capacity=10000,
        batch_size=8,
        num_updates=1000000,
        online_steps=1000,
        log_interval=10,
        save_dir="F:/MineRL/VPT/model_save",
        model="F:/MineRL/VPT/foundation-model-1x.model",
        iql_weights="F:/MineRL/VPT/model_save/iql_agent.pth",
        video_interval=10,
        video_dir="F:/MineRL/VPT/videos",
        bc_loss_coef=0.1,
        critic_tau=0.005,
        eval_interval=2000,  # 评估间隔
        online_start_update=10000,  # 开始在线训练的更新步数
        online_batch_size=16,  # 在线训练的批量大小
        clean_memory_interval=1000,  # 清理显存间隔
    )
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)