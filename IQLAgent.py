import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as T
import copy
from torch.optim.lr_scheduler import CyclicLR
import torchvision.models as models
import swanlab
ACTION_KEYS = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sprint', 'camera']
# 预处理图像的目标分辨率 (H, W)
RESOLUTION = (128, 128)
# 对于 camera 动作，这里假设输出为二维连续值
CAMERA_DIM = 2
trans_pipeline = T.Compose([
        T.ToPILImage(),
        T.Resize((72,128)),
        T.ToTensor()
    ])
# 离散动作数量（除 camera 外）
# 新增离散动作映射：这里定义了键盘动作与离散化 camera 动作
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
# 更新离散动作数量
NUM_DISCRETE = len(DISCRETE_ACTIONS)
import os
def process_cave_features(self, latent, cave_prob):
    if cave_prob is not None:
        # 标准化处理
        if not isinstance(cave_prob, torch.Tensor):
            cave_prob = torch.tensor(cave_prob, dtype=torch.float32, device=self.device)
        if cave_prob.dim() == 1:
            cave_prob = cave_prob.view(-1, 1)
            
        # 平滑处理cave影响，避免硬阈值
        cave_attention = torch.sigmoid(5 * (cave_prob - 0.5))  # 平滑S形曲线
        cave_features = self.cave_embedding(cave_prob)
        
        # 使用注意力机制融合
        gated_features = cave_features * cave_attention
        latent = latent * (1 + 0.2 * gated_features)  # 乘性影响更自然
    
    return latent
def convert_onehot_to_action_dict(one_hot):
    
    """
    将11维one-hot向量转换为动作字典，对应索引定义与Train.py一致：
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
    """
    mapping = {
         0: {'forward': 1},
         1: {'back': 1},
         2: {'left': 1},
         3: {'right': 1},
         4: {'camera': [10, 0]},    # Look up
         5: {'camera': [-10, 0]},   # Look down
         6: {'camera': [0, -10]},   # Look left
         7: {'camera': [0, 10]},    # Look right
         8: {'forward': 1, 'jump': 1},
         9: {'jump': 1},
         10: {'attack': 1}
    }

    return mapping.get(one_hot, {'forward': 1})

def soft_maximum(x, tau=1.0):
    """
    计算软最大值
    x: 输入张量
    tau: 温度参数，控制平滑程度（越小越接近真实最大值）
    """
    # 使用 log-sum-exp 技巧避免数值溢出
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x_exp = torch.exp((x - x_max) / tau)
    return x_max + tau * torch.log(torch.sum(x_exp, dim=1))
#MOVE_ACTION_INDICES = torch.tensor([ACTION_KEYS.index(a) for a in MOVE_ACTIONS], device='cuda')
class IQLearnAgent(nn.Module):
    def __init__(self, env=None, args=None):
        """
        env: 训练环境，可用于获取 action_space 等信息（若使用字典格式的动作，则按 ACTION_KEYS 构造）
        args: 配置参数，至少需要包含 device, gamma,
              可选 expert_loss_coef 指定专家数据损失权重（默认 1.0）
        """
        super(IQLearnAgent, self).__init__()
        self.total_updates = 0
        
        self.device = torch.device(args.device) if args and hasattr(args, "device") else torch.device("cpu")
        self.gamma = args.gamma if args and hasattr(args, "gamma") else 0.99
        self.expert_loss_coef = args.expert_loss_coef if args and hasattr(args, "expert_loss_coef") else 1.0
        
        self.max_updates = args.num_updates if args and hasattr(args, "num_updates") else 1000000
        self.warmup_steps = args.warmup_steps if args and hasattr(args, "warmup_steps") else 1000000
       
        self.bc_weight = 0.9  # 固定BC权重
       
        self.max_advantage = 100.0  # 限制优势值范围
        # CNN 特征提取
        mobilenet = models.mobilenet_v3_small(pretrained=True,weights='DEFAULT').to(self.device)
        # 取出 features 部分
        self.feature_extractor = nn.Sequential(*list(mobilenet.features.children())[:7])
        # 冻结部分参数，或者全部微调，根据需要决定
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        # 注册 Grad-CAM hook，在最后一层卷积上获取激活和梯度
        self.activations = None
        self.gradients = None
        last_conv = self.feature_extractor[-1]
        last_conv.register_forward_hook(self.save_activation)
        last_conv.register_full_backward_hook(self.save_gradient)

        # 测试一下输出维度（输入分辨率为 RESOLUTION=(128,128)）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *RESOLUTION).to(self.device)
            features = self.feature_extractor(dummy)
            self.flatten_dim = features.view(1, -1).shape[1]
        
       # 定义cave特征的嵌入维度
        self.cave_embed_dim = 512  # 与hidden_dim保持一致

        # 通过 LSTM 融合时序信息，注意输入维度需要考虑cave_features
        # 修改这里，接受组合后的维度
        self.temporal_rnn = nn.LSTM(
            input_size=self.flatten_dim + self.cave_embed_dim, 
            hidden_size=512, 
            batch_first=True
        )

        self.hidden_dim = 512
        # Actor 部分：输出离散动作 logits（camera 动作由单独回归）
        self.discrete = nn.Sequential(
            nn.Linear(self.hidden_dim, 512), nn.ReLU(),
            nn.Linear(512, NUM_DISCRETE)
        )
        # Critic 分支 2：用状态和动作拼接后输入，输出一个标量 Q 值（用于 critic_fn）
        self.critic_sa = nn.Sequential(
            nn.Linear(self.hidden_dim + NUM_DISCRETE , 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.tau = 0.005
        self.critic_target = copy.deepcopy(self.critic_sa)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        # BC损失权重
        self.bc_loss_coef = args.bc_loss_coef if args and hasattr(args, "bc_loss_coef") else 1.0
        
        self.hidden_state = None
        # 定义对应的优化器
        # 图像预处理：使用 torchvision.transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(RESOLUTION),
            T.ToTensor()
        ])
        
        self.video_dir = args.video_dir if args and hasattr(args, "video_dir") else "videos"
       # cave_embedding层的定义也要相应修改
        self.cave_embedding = nn.Linear(1, self.cave_embed_dim)
        # 定义α的衰减区间，从 1e-1 衰减到 1e-2
        self.alpha_start = 1e-1
        self.alpha_final = 1e-2
        self.alpha = self.alpha_start
    def reset_memory(self, batch_size=1):
        device = self.device
        self.hidden_state = (torch.zeros(1, batch_size, self.hidden_dim, device=device),
                             torch.zeros(1, batch_size, self.hidden_dim, device=device))
    
    def extract_features(self, img_tensor):
        """统一处理输入张量维度，并展平特征"""
        # 确保输入是4D张量 [B, C, H, W]
        if img_tensor.dim() == 5:  # [B, 1, C, H, W]
            img_tensor = img_tensor.squeeze(1)
        elif img_tensor.dim() == 3:  # [C, H, W]
            img_tensor = img_tensor.unsqueeze(0)

        if img_tensor.shape[1] != 3:
            print(f"Warning: Unexpected input shape: {img_tensor.shape}")
            if img_tensor.shape[2] == 3:
                img_tensor = img_tensor.permute(0, 3, 1, 2)
        
        assert img_tensor.dim() == 4 and img_tensor.shape[1] == 3, \
            f"Invalid input tensor shape: {img_tensor.shape}, expected [B, 3, H, W]"
        
        features = self.feature_extractor(img_tensor)  # [B, C_out, H_out, W_out]
        # 将 CNN 输出展平为 [B, flatten_dim]
        # 增强对图像暗区域的感知(矿洞通常较暗)
        features_mean = features.mean(dim=1, keepdim=True)
        dark_attention = 1.0 - torch.sigmoid(5 * (features_mean - 0.3))
        features = features * (1 + 0.3 * dark_attention)
        
        return features.view(features.size(0), -1)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def update_alpha(self):
        """
        根据训练步数 total_updates，线性衰减α，从 alpha_start 到 alpha_final
        """
        ratio = min(float(self.total_updates) / self.max_updates, 1.0)
        self.alpha = self.alpha_start - ratio * (self.alpha_start - self.alpha_final)
        self.total_updates += 1  # 添加这一行
        # 可选：记录当前alpha值
        if self.total_updates % 100 == 0:
            print(f"更新 {self.total_updates}: alpha = {self.alpha:.4f}, temperature = {self.alpha * 5.0:.4f}")
    def compute_gradcam(self, input_image, target_action_index):
        was_training = self.training
        # 临时切换到训练模式以保证梯度计算，但尽量避免多次保留计算图
        self.train()
        try:
            self.zero_grad()
            outputs = self.forward(input_image)
            discrete = outputs[0] if (isinstance(outputs, tuple) and len(outputs) >= 2) else outputs

            # 计算target对activations的梯度，若后续不需要多次反向传播，可将 retain_graph 设为 False
            target = discrete[0, target_action_index]
            grads = torch.autograd.grad(target, self.activations, retain_graph=False)[0]
            self.gradients = grads  # 保存此次计算的梯度

            if self.activations is None or self.gradients is None:
                return torch.zeros((1, 1, input_image.shape[2], input_image.shape[3]),
                                device=input_image.device)

            activations = self.activations.detach()  # 释放计算图
            gradients = self.gradients.detach()
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
            # 清空hook保存的临时变量，避免持续占用显存
            self.activations = None
            self.gradients = None
            return cam
        finally:
            if not was_training:
                self.eval()
    def forward(self, x, cave_prob=None):
        features = self.extract_features(x)  # [B, flatten_dim]
        
        # 在LSTM前融合cave_prob
        if cave_prob is not None:
            if not isinstance(cave_prob, torch.Tensor):
                cave_prob = torch.tensor(cave_prob, dtype=torch.float32, device=self.device)
            if cave_prob.dim() == 1:
                cave_prob = cave_prob.view(-1, 1)
                
            # 生成cave特征
            cave_features = self.cave_embedding(cave_prob)  # [B, cave_embed_dim]
            
            # 将cave特征与图像特征拼接
            combined_features = torch.cat([features, cave_features.squeeze(-1)], dim=1)
        else:
            combined_features = features
        
        # 使用拼接后的特征输入LSTM
        features_seq = combined_features.unsqueeze(1)  # [B, 1, combined_dim]
        lstm_out, self.hidden_state = self.temporal_rnn(features_seq, self.hidden_state)
        latent = lstm_out.squeeze(1)  # [B, hidden_dim]
        
        # 后续处理
        discrete = self.discrete(latent)
        return discrete
    def compute_V(self, obs, cave_prob=None, target=False):
        """计算状态值函数V(s) - 使用正确的IQ-Learn期望计算方法
        
        V(s) = E_a~π [Q(s,a)] = ∑_a π(a|s)·Q(s,a)
        """
        # 获取所有动作的概率
        logits = self.forward(obs, cave_prob)  # [B, NUM_DISCRETE]
        probs = F.softmax(logits, dim=-1)
        
        # 初始化V值
        batch_size = obs.shape[0]
        v_values = torch.zeros((batch_size, 1), device=self.device)
        
        # 对每个可能的动作计算Q值并累加
        for a in range(NUM_DISCRETE):
            # 创建one-hot动作表示
            actions = F.one_hot(torch.full((batch_size,), a, device=self.device, dtype=torch.long), 
                            num_classes=NUM_DISCRETE).float()
            # 根据target参数选择网络
            q_values = self.critic_fn(obs, actions, cave_prob, target=target)
            v_values += probs[:, a:a+1] * q_values
                
        return v_values
    def get_action(self, obs, cave_prob, exploration=True):
        """使用一致的基于温度的采样策略选择动作"""
        with torch.no_grad():
            # 图像预处理
            image = self.transform(obs["pov"]).to(self.device)
            image = image.unsqueeze(0)
            if not isinstance(cave_prob, torch.Tensor):
                cave_prob = torch.tensor([[cave_prob]], dtype=torch.float32, device=self.device)
            
            # 获取动作logits
            logits = self.forward(image, cave_prob)
            
            # 根据exploration调整温度
            temperature = self.alpha * (5.0 if exploration else 1.0)
            
            # 环境因素影响温度
            if cave_prob > 0.5 and exploration:
                temperature *= 1.2  # 在洞穴中适度增加随机性
            
            if exploration:
                # 使用Gumbel-Softmax实现entropy-regularized采样
                action_onehot = F.gumbel_softmax(logits, tau=temperature, hard=True)
                action_idx = torch.argmax(action_onehot, dim=-1).item()
                
                # 记录熵值
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                    if hasattr(self, 'total_updates') and self.total_updates % 100 == 0:
                        print(f"Action entropy: {entropy:.4f}, Temperature: {temperature:.2f}")
            else:
                action_idx = torch.argmax(logits, dim=-1).item()
                
            action = convert_onehot_to_action_dict(action_idx)
            return action

    def critic_fn(self, obs, actions, cave_prob=None, target=False):
        """拼接动作之后使用critic_sa/critic_target直接给出动作Q值"""
        batch_size = obs.shape[0]
        # 重置LSTM状态
        self.reset_memory(batch_size)
        # 特征提取与时序融合获得 latent 表征
        features = self.extract_features(obs)
        cave_features = self.cave_embedding(cave_prob)  # 使用相同的embedding层
        features = torch.cat([features, cave_features], dim=1)  # 拼接特征
        features_seq = features.unsqueeze(1)
        lstm_out, _ = self.temporal_rnn(features_seq, self.hidden_state)
        latent = lstm_out.squeeze(1).clone()  # 避免就地操作
        
        if cave_prob is not None:
            if not isinstance(cave_prob, torch.Tensor):
                cave_prob = torch.tensor(cave_prob, dtype=torch.float32, device=self.device)
            if cave_prob.dim() == 1:
                cave_prob = cave_prob.view(-1, 1)
            process_cave_features(self, latent, cave_prob)
            
        # 确保actions是正确的形状
        if actions.dim() == 1:
            actions = F.one_hot(actions.long(), num_classes=NUM_DISCRETE).float()
        
        combined_input = torch.cat([latent, actions], dim=1).clone()
        
        # 根据target参数选择网络
        if target:
            q_val = self.critic_target(combined_input)
        else:
            q_val = self.critic_sa(combined_input)
            
        return q_val


    def soft_update(self, net, target_net):
     
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
    def compute_iql_loss(self, batch):
        """按照IQ-Learn算法计算损失
        
        IQ-Learn的三个关键组成部分:
        1. 最大化专家行为的恢复奖励
        2. 使专家Q值超过非专家Q值
        3. 满足贝尔曼方程
        """
        obs, next_obs, actions, rewards, dones, is_expert, cave_probs, next_cave_probs,weights = batch
        batch_size = obs.shape[0]
        if len(batch) > 8:
            obs, next_obs, actions, rewards, dones, is_expert, cave_probs, next_cave_probs, weights = batch[:9]
            has_weights = True
        else:
            obs, next_obs, actions, rewards, dones, is_expert, cave_probs, next_cave_probs = batch
            # 如果没有权重，使用全1权重
        weights = torch.ones_like(rewards)
        has_weights = False
        # 重置LSTM状态
        self.reset_memory(batch_size)
        
        # 1. 计算当前Q值
        current_q = self.critic_fn(obs, actions, cave_probs)  # [B,1]
        
        # 2. 计算下一状态的V值 (使用目标网络)
        with torch.no_grad():
            next_v = self.compute_V(next_obs, next_cave_probs, target=True)
            target_q = rewards + self.gamma * (1 - dones) * next_v
        td_errors = current_q.detach() - target_q
        # 3. 计算恢复的奖励 (IQ-Learn的核心)
        recovered_rewards = current_q - self.gamma * next_v
        swanlab.log({"recovered_rewards": recovered_rewards.mean().item()})
        
        # 4. 分离专家数据和在线数据
        expert_mask = is_expert.squeeze()
        
        # 初始化损失组件
        expert_loss = torch.tensor(0.0, device=self.device)
        margin_loss = torch.tensor(0.0, device=self.device)
        
        # 5. 计算专家损失 - 最大化专家行为的恢复奖励
        if expert_mask.any():
                
            # 应用权重到专家损失
            expert_loss = -(recovered_rewards[expert_mask] ).mean()
            swanlab.log({"expert_loss": expert_loss.item()})
            
        
                    
        # 6. 计算间隔损失 - 确保专家Q值高于非专家Q值
        if (~expert_mask).any():
                expert_q = current_q[expert_mask]
                non_expert_q = current_q[~expert_mask]
                
                # 正确方向: 专家Q值应当高于非专家Q值
                margin_loss = F.relu(non_expert_q.mean() - expert_q.mean() + 2.0)
                swanlab.log({"margin_loss": margin_loss.item()})
        
        # 7. 贝尔曼损失 - 确保Q值满足贝尔曼方程
        bellman_loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        swanlab.log({"bellman_loss": bellman_loss.item()})
        
        # 8. 组合Critic损失
        critic_loss = expert_loss + margin_loss + 0.5 * bellman_loss
        
        # 9. Actor损失 - 最大化预期Q值和熵正则化
        action_logits = self.forward(obs, cave_probs)
        probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # 计算每个动作的Q值期望
        q_values = torch.zeros_like(probs)
        for a in range(NUM_DISCRETE):
            actions_a = F.one_hot(torch.full((batch_size,), a, device=self.device, dtype=torch.long), 
                                num_classes=NUM_DISCRETE).float()
            q_a = self.critic_fn(obs, actions_a, cave_probs).squeeze()
            q_values[:, a] = q_a
        
        # 策略梯度: 最大化 E_a~π[Q(s,a)] + α·H(π)
        policy_loss = -(probs * q_values).sum(dim=1).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()  # 计算正熵
        swanlab.log({"entropy": entropy.item()})
        
        self.update_alpha()  # 更新熵系数
        actor_loss = policy_loss - self.alpha * entropy
        
        return critic_loss, actor_loss

    
    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def save(self, path):
        torch.save(self.state_dict(), path)
