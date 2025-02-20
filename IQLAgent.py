import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as T
import copy
from torch.optim.lr_scheduler import CyclicLR

import swanlab
ACTION_KEYS = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint', 'use', 'drop', 'inventory', 'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5', 'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9', 'camera']
# 预处理图像的目标分辨率 (H, W)
RESOLUTION = (128, 128)
# 对于 camera 动作，这里假设输出为二维连续值
CAMERA_DIM = 2
# 离散动作数量（除 camera 外）
NUM_DISCRETE = len(ACTION_KEYS) - 1
import os
MOVE_ACTIONS = ['forward', 'back', 'left', 'right', 'jump', 'camera']
#MOVE_ACTION_INDICES = torch.tensor([ACTION_KEYS.index(a) for a in MOVE_ACTIONS], device='cuda')
class IQLearnAgent(nn.Module):
    def __init__(self, env=None, args=None):
        """
        env: 训练环境，可用于获取 action_space 等信息（若使用字典格式的动作，则按 ACTION_KEYS 构造）
        args: 配置参数，至少需要包含 device, gamma,
              可选 expert_loss_coef 指定专家数据损失权重（默认 1.0）
        """
        super(IQLearnAgent, self).__init__()
        self.device = torch.device(args.device) if args and hasattr(args, "device") else torch.device("cpu")
        self.gamma = args.gamma if args and hasattr(args, "gamma") else 0.99
        self.expert_loss_coef = args.expert_loss_coef if args and hasattr(args, "expert_loss_coef") else 1.0
        self.MOVE_ACTION_INDICES = torch.tensor(
            [ACTION_KEYS.index(a) for a in MOVE_ACTIONS], 
            device=self.device
        )
        self.alpha = 0.1  # 初始温度参数
        self.min_alpha = 0.01  # 最小温度
        self.alpha_decay = (0.1 - 0.01) / args.num_updates  # 线性衰减率
        
        # CNN 特征提取，与 PPO 模型类似
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *RESOLUTION)
            self.flatten_dim = self.feature_extractor(dummy).shape[1]
        # 通过 LSTM 融合时序信息（模仿 PPO 结构）
        self.temporal_rnn = nn.LSTM(input_size=self.flatten_dim, hidden_size=256, batch_first=True)
        self.hidden_dim = 256
        # Actor 部分：输出离散动作 logits（camera 动作由单独回归）
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, 512), nn.ReLU(),
            nn.Linear(512, NUM_DISCRETE)
        )
        # Critic 部分：输出离散动作对应的 Q 值
         # Critic 分支 1：仅用 latent 来估计每个离散动作的 Q 值（例如用于 actor 分支）
        self.critic_discrete = nn.Sequential(
            nn.Linear(self.hidden_dim, 512), nn.ReLU(),
            nn.Linear(512, NUM_DISCRETE)
        )
        
        # Critic 分支 2：用状态和动作拼接后输入，输出一个标量 Q 值（用于 critic_fn）
        self.critic_sa = nn.Sequential(
            nn.Linear(self.hidden_dim + (NUM_DISCRETE + CAMERA_DIM), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.tau = 0.005
        self.critic_target = copy.deepcopy(self.critic_sa)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        # BC损失权重
        self.bc_loss_coef = args.bc_loss_coef if args and hasattr(args, "bc_loss_coef") else 1.0
        # Camera 动作回归头
        self.camera_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512), nn.ReLU(),
            nn.Linear(512, CAMERA_DIM)
        )
        self.hidden_state = None
         # 定义对应的优化器
        # 图像预处理：使用 torchvision.transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(RESOLUTION),
            T.ToTensor()
        ])
        self.video_dir = args.video_dir if args and hasattr(args, "video_dir") else "videos"
        # 新增：将 cave_prob 映射到与 hidden_dim 相同的向量
        self.cave_embedding = nn.Linear(1, self.hidden_dim)
    def update_alpha(self, step):
        """更新温度参数"""
        self.alpha = max(self.min_alpha, 
                        0.1 - step * self.alpha_decay)
      
    def reset_memory(self, batch_size=1):
        device = self.device
        self.hidden_state = (torch.zeros(1, batch_size, self.hidden_dim, device=device),
                             torch.zeros(1, batch_size, self.hidden_dim, device=device))
    
    def extract_features(self, img_tensor):
        # img_tensor: [B, 3, H, W] (H, W = RESOLUTION)
        return self.feature_extractor(img_tensor)
    
    def forward(self, x,cave_prob=None):
        features = self.extract_features(x)  # [B, flatten_dim]
        features_seq = features.unsqueeze(1)   # [B, 1, flatten_dim]
        lstm_out, self.hidden_state = self.temporal_rnn(features_seq, self.hidden_state)
        latent = lstm_out.squeeze(1)           # [B, hidden_dim]
        if cave_prob is not None:
            # 如果 cave_prob 不是 Tensor，则转换为 Tensor
            if not isinstance(cave_prob, torch.Tensor):
                cave_prob = torch.tensor(cave_prob, dtype=torch.float32, device=self.device)
            # 假定 cave_prob 的 shape 为 [B] 或 [B,1]
            if cave_prob.dim() == 1:
                cave_prob = cave_prob.view(-1, 1)
            cave_emb = self.cave_embedding(cave_prob)  # [B, hidden_dim]
            latent = latent + cave_emb
    # 离散动作部分：直接通过 critic_discrete 得到每个动作对应的 Q 值
        action_logits = self.actor(latent)     # [B, NUM_DISCRETE]
        q_values = self.critic_discrete(latent)  # [B, NUM_DISCRETE]
        # camera 部分（连续回归输出）
        camera_output = self.camera_head(latent) # [B, CAMERA_DIM]
        return action_logits, q_values, camera_output

    def get_action(self, obs,cave_prob ,exploration=True):
        """
        根据输入观测选择动作。
        obs: 字典，要求包含 "pov" 键，其形状为 (H, W, 3)
        返回：动作字典，格式为：
            离散动作部分：one-hot 表示（基础键见 ACTION_KEYS[:-1]）
            连续 camera 动作：回归输出
        """
        image = self.transform(obs["pov"]).to(self.device)  # [3, H, W]
        image = image.unsqueeze(0)  # [1, 3, H, W]
        
        with torch.no_grad():
            action_logits, q_values, camera_output = self(image,cave_prob)
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
        discrete_keys = ACTION_KEYS[:-1]
        action = {key: 0 for key in discrete_keys}
        chosen_key = discrete_keys[action_idx]
        action[chosen_key] = 1
        action["camera"] = camera_output.squeeze(0).cpu().numpy().tolist()
        return action


    def flatten_action(self, action):
        """
        将动作字典转换为展平的 numpy 数组，以适配模型输入。
        action: 包含离散动作和 camera 连续动作的字典或列表
        返回: np.ndarray, 形状为 (NUM_DISCRETE + CAMERA_DIM,)
        """
        discrete_keys = ACTION_KEYS[:-1]
        
        # 如果已经是 Tensor，直接返回
        if isinstance(action, torch.Tensor):
            #print("tensorshape:",action.shape)
            return action
        # 如果是列表，则逐个处理后拼接
        if isinstance(action, list):
            flattened_list = [self.flatten_action(act) for act in action]
            #print("listshape:",torch.stack(flattened_list, dim=0).shape)
            return torch.stack(flattened_list, dim=0).squeeze(1)
        # 如果是字典，则处理为 tensor
        if isinstance(action, dict):
            #print("action:",action)
            discrete_keys = ACTION_KEYS[:-1]
            discrete_values = []
            for key in discrete_keys:
                val = action[key]
                if isinstance(val, np.ndarray):
                    # 将 numpy 数组 squeeze 成标量
                    val = np.array(val).squeeze()
                    discrete_values.append(float(val))
                else:
                    discrete_values.append(float(val))
            discrete_part = np.array(discrete_values, dtype=np.float32)
            # 处理 camera 部分
           # print("discrete_part:",discrete_part.shape)
            camera_part = np.array(action.get("camera", [0.0, 0.0]), dtype=np.float32).flatten()
          #  print("camera_part:",camera_part.shape)
            flat = np.concatenate([discrete_part, camera_part])
          #  print("dictshape:",torch.tensor(flat, dtype=torch.float32, device=self.device).unsqueeze(0).shape)
            return torch.tensor(flat, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _get_action_log_prob(self, obs, expert_actions):
        """
        计算专家动作的 BC loss，其中 expert_actions 为形状 [B, NUM_DISCRETE+CAMERA_DIM] 的 tensor
        """
        # 如果 expert_actions 是列表，则转换为 tensor
        if isinstance(expert_actions, list):
            expert_actions = torch.stack(expert_actions, dim=0)
        # 如果 expert_actions 额外多了一个维度（例如 [B,1,22]），则 squeeze 掉
        if expert_actions.dim() == 3 and expert_actions.size(1) == 1:
            expert_actions = expert_actions.squeeze(1)
        
        # 提取 latent 表征，计算离散动作 logits 以及 camera 部分预测
        latent = self._get_latent(obs)  # shape: [B, hidden_dim]
        discrete_logits = self.actor(latent)  # [B, NUM_DISCRETE]
        camera_pred = self.camera_head(latent)  # [B, CAMERA_DIM]
        
        # 确保 expert_actions 与 latent 在同一设备上
        expert_actions = expert_actions.to(latent.device)
        
        # expert_actions 拆分成离散部分和 camera 部分
        expert_discrete = expert_actions[:, :NUM_DISCRETE]  # [B, NUM_DISCRETE]
        expert_camera = expert_actions[:, NUM_DISCRETE:]    # [B, CAMERA_DIM]
        
        # 对离散部分计算交叉熵损失（假设 expert_discrete 为 one-hot 编码）
        log_probs = F.log_softmax(discrete_logits, dim=1)  # [B, NUM_DISCRETE]
        discrete_loss = - (expert_discrete * log_probs).sum(dim=1).mean()
        
        # 对 camera 部分计算均方误差
        camera_loss = F.mse_loss(camera_pred, expert_camera)
        
        total_loss = discrete_loss + self.bc_loss_coef * camera_loss
        return total_loss
    def critic_fn(self, obs, action,cave_prob=None):
        # 统一action维度
        batch_size = obs.shape[0]
        if len(action.shape) > 2:
            action = action.view(batch_size, -1)
        
        # 重置LSTM状态
        self.reset_memory(batch_size)
        
        # 特征提取
        features = self.extract_features(obs)
        features_seq = features.unsqueeze(1)
        
        # LSTM处理
        lstm_out, _ = self.temporal_rnn(features_seq, self.hidden_state)
        latent = lstm_out.squeeze(1)
        if cave_prob is not None:
            if cave_prob.dim() == 1:
                cave_prob = cave_prob.view(-1, 1)
            latent = latent + self.cave_embedding(cave_prob)
        combined = torch.cat([latent, action], dim=1)
        return self.critic_sa(combined)
    def _get_latent(self, obs, cave_prob=None):
        # 重新根据当前 batch_size 重置隐藏状态
        batch_size = obs.shape[0]
        self.reset_memory(batch_size=batch_size)
        features = self.extract_features(obs)         # [B, self.flatten_dim]
        features_seq = features.unsqueeze(1)            # [B, 1, self.flatten_dim]
        lstm_out, _ = self.temporal_rnn(features_seq, self.hidden_state)
        latent = lstm_out.squeeze(1)                    # [B, hidden_dim]
        # 如果传入 cave_prob，则将其通过线性映射后与 latent 相加
        if cave_prob is not None:
            if cave_prob.dim() == 1:
                cave_prob = cave_prob.view(-1, 1)
            latent = latent + self.cave_embedding(cave_prob)
        return latent

    def getV(self, obs, cave_prob=None):
        """
        计算状态 V 值。在利用当前状态 obs 提取 latent
        表征时，同时融合 cave_prob 信息。
        """
        batch_size = obs.shape[0]
        # 提取 latent 表征，同时传入 cave_prob
        latent = self._get_latent(obs, cave_prob)
        # 计算 camera 部分动作
        camera_actions = self.camera_head(latent)  # [B, CAMERA_DIM]
        camera_actions = camera_actions.unsqueeze(1).repeat(1, NUM_DISCRETE, 1)  # [B, NUM_DISCRETE, CAMERA_DIM]
        # 构造离散动作 one-hot 表示
        discrete_actions = torch.eye(NUM_DISCRETE, device=obs.device)  # [NUM_DISCRETE, NUM_DISCRETE]
        discrete_actions = discrete_actions.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, NUM_DISCRETE, NUM_DISCRETE]
        # 拼接离散和连续动作部分
        actions = torch.cat([discrete_actions, camera_actions], dim=-1)  # [B, NUM_DISCRETE, NUM_DISCRETE+CAMERA_DIM]
        actions = actions.view(-1, NUM_DISCRETE + CAMERA_DIM)  # [B*NUM_DISCRETE, NUM_DISCRETE+CAMERA_DIM]
        
        # 需要将 latent 形状重复 NUM_DISCRETE 次
        latent = latent.unsqueeze(1).repeat(1, NUM_DISCRETE, 1)  # [B, NUM_DISCRETE, hidden_dim]
        latent = latent.view(-1, self.hidden_dim)  # [B*NUM_DISCRETE, hidden_dim]
        
        # 拼接后计算 Q 值
        combined = torch.cat([latent, actions], dim=1)  # [B*NUM_DISCRETE, hidden_dim+NUM_DISCRETE+CAMERA_DIM]
        q_values = self.critic_sa(combined).view(batch_size, NUM_DISCRETE)  # [B, NUM_DISCRETE]
        # 取每个状态下最大的 Q 值作为 V(s)
        V, _ = torch.max(q_values, dim=1, keepdim=True)  # [B, 1]
        return V
    def soft_update(self, net, target_net):
        ## 延迟update，EMA update，每次 update 很少的部分的
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
    def compute_iql_loss(self, batch):
        def soft_maximum(q_values):
            probs = F.softmax(q_values / self.alpha, dim=1)
            return (probs * q_values).sum(dim=1)
           
        
        obs, next_obs, actions, rewards, dones, is_expert, cave_probs = batch
        current_Q = self.critic_fn(obs, actions, cave_probs)
        with torch.no_grad():
            next_latent = self._get_latent(next_obs, cave_probs)
            next_actions = self.actor(next_latent)
            # 使用与critic_fn相同的处理方式
            # 修复：需要正确处理action维度
            camera_pred = self.camera_head(next_latent)  # [B, CAMERA_DIM]
            next_full_actions = torch.cat([next_actions, camera_pred], dim=1)  # [B, NUM_DISCRETE + CAMERA_DIM]
            combined = torch.cat([next_latent, next_full_actions], dim=1)
            next_Q = self.critic_target(combined)
            next_V = self.getV(next_obs, cave_probs)
            target_Q = rewards + (1 - dones) * self.gamma * torch.min(next_Q, next_V)
            
        
        online_mask = ~is_expert.squeeze()
        expert_mask = is_expert.squeeze()
        loss_dict = {}
        total_loss = 0.0
        # 在 IQLAgent.py 的 compute_iql_loss 中添加：
        print(f"[Debug] current_Q: {current_Q.mean().item():.2f} ± {current_Q.std().item():.2f}")
        print(f"[Debug] target_Q: {target_Q.mean().item():.2f} ± {target_Q.std().item():.2f}")
        # 在线数据损失
        if online_mask.sum() > 0:
            online_loss = F.mse_loss(current_Q[online_mask], target_Q[online_mask])
            total_loss += online_loss
            loss_dict['online_loss'] = online_loss.item()
        else:
            loss_dict['online_loss'] = 0.0

        # 专家数据损失
        if expert_mask.sum() > 0:
            expert_Q = current_Q[expert_mask]
            expert_V = soft_maximum(self.critic_discrete(self._get_latent(obs[expert_mask])))
            expert_loss = F.mse_loss(expert_Q, expert_V)
            total_loss += self.expert_loss_coef * expert_loss
            loss_dict['expert_loss'] = expert_loss.item()

            # 计算 BC 损失
            # 注意：这里 expert_actions 需要从 actions 中正确取出
            expert_actions = [actions[i] for i in torch.where(expert_mask)[0]]
            bc_loss = self._get_action_log_prob(obs[expert_mask], expert_actions)
            total_loss += self.bc_loss_coef * bc_loss
            loss_dict['bc_loss'] = bc_loss.item()
        else:
            loss_dict['expert_loss'] = 0.0
            loss_dict['bc_loss'] = 0.0

        self.soft_update(self.critic_sa, self.critic_target)
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

    
    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def save(self, path):
        torch.save(self.state_dict(), path)
    def _save_video(self, frames, ep):
        import os
        if not frames:
            print(f"警告：episode {ep} 无视频帧可保存。")
            return
        height, width, _ = frames[0].shape
        print(f"视频帧尺寸: {width}x{height}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   
        video_path = os.path.join(self.video_dir, f"episode_{ep}.mp4")
        print(f"尝试保存视频至: {video_path}")
        writer = cv2.VideoWriter(video_path, fourcc, 20, (width, height))
        if not writer.isOpened():
            print("错误：无法打开 VideoWriter。")
            return
        print(len(frames))
        writer.release()
        print(f"视频已保存至：{video_path}")