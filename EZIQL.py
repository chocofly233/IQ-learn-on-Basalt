# 验证IQL智能体在简单环境中的表现
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from collections import deque
import random
import copy

# 重构版SimpleIQLAgent（符合原始IQ-Learn论文）
class SimpleIQLAgent(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super().__init__()
        self.device = args.device
        self.gamma = args.gamma
        self.tau = args.critic_tau
        self.alpha_start = 1e-1
        self.alpha_final = 1e-2
        self.alpha = self.alpha_start
        self.max_updates = args.num_updates if args and hasattr(args, "num_updates") else 1000000

        self.total_updates = 0
        self.alpha = args.init_alpha
        self.action_dim = action_dim
        
        # 简单网络结构
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # 状态动作价值函数
        self.critic_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 目标网络
        self.critic_target = copy.deepcopy(self.critic_sa)
        for param in self.critic_target.parameters():
            param.requires_grad = False
    
    def forward(self, state):
        return self.actor(state)
    def update_alpha(self):
        """根据训练进度动态调整温度参数"""
        ratio = min(float(self.total_updates) / self.max_updates, 1.0)
        self.alpha = self.alpha_start - ratio * (self.alpha_start - self.alpha_final)
        self.total_updates += 1
    def get_action(self, state, exploration=True):
        """选择动作，支持探索，使用Gumbel-Softmax采样"""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(state)
            
            if exploration:
                # 使用与IQLearnAgent一致的温度策略
                temperature = max(0.5, 1.0 - 0.0005 * self.total_updates)
                
                # 使用Gumbel-Softmax采样，与IQLearnAgent保持一致
                action_probs = F.gumbel_softmax(logits, tau=temperature, hard=True)
                action = torch.argmax(action_probs, dim=-1).item()
                
                # 记录熵值用于调试
                if self.total_updates % 100 == 0:
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                    print(f"Action entropy: {entropy:.4f}, Temperature: {temperature:.2f}")
            else:
                action = torch.argmax(logits, dim=-1).item()
                
            return action
    
    def critic_fn(self, state, action, target=False):
        """评估状态动作对的Q值"""
        # 处理动作输入
        if isinstance(action, int) or (isinstance(action, torch.Tensor) and action.dim() == 0):
            action = torch.tensor([action], device=self.device)
            
        if action.dim() == 1:
            action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        else:
            action_onehot = action
            
        # 连接状态和动作
        sa_input = torch.cat([state, action_onehot], dim=-1)
        
        # 使用目标网络或在线网络
        if target:
            return self.critic_target(sa_input)
        else:
            return self.critic_sa(sa_input)
    
    def compute_V(self, state, target=False):
        """计算状态值函数V(s)
        
        IQ-Learn使用下式计算V值:
        V(s) = E_a~π [Q(s,a)] = ∑_a π(a|s)·Q(s,a)
        """
        # 获取所有动作的概率
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        # 初始化V值
        batch_size = state.shape[0]
        v_values = torch.zeros((batch_size, 1), device=self.device)
        
        # 对每个可能的动作计算Q值并累加
        for a in range(self.action_dim):
            # 创建one-hot动作
            actions = torch.full((batch_size,), a, device=self.device, dtype=torch.long)
            q_values = self.critic_fn(state, actions, target=target)
            v_values += probs[:, a:a+1] * q_values
            
        return v_values
    
    def compute_losses(self, batch):
        """按照IQ-Learn算法计算损失
        
        IQ-Learn的三个关键组成部分:
        1. 最大化专家行为的恢复奖励
        2. 使专家Q值超过非专家Q值
        3. 满足贝尔曼方程
        """
        state, next_state, action, reward, done, is_expert = batch
        
        # 1. 计算当前Q值
        current_q = self.critic_fn(state, action)
        
        # 2. 计算下一状态的V值 (使用目标网络)
        with torch.no_grad():
            next_v = self.compute_V(next_state, target=True)
            target_q = reward + self.gamma * (1 - done) * next_v
        
        # 3. 计算恢复的奖励 (IQ-Learn的核心)
        recovered_rewards = current_q - self.gamma * next_v
        
        # 4. 分离专家数据和在线数据
        expert_mask = is_expert.squeeze()
        expert_indices = torch.where(expert_mask)[0] if expert_mask.any() else None
        non_expert_indices = torch.where(~expert_mask)[0] if (~expert_mask).any() else None
        
        # 初始化损失组件
        expert_loss = torch.tensor(0.0, device=self.device)
        margin_loss = torch.tensor(0.0, device=self.device)
        
        # 5. 计算专家损失 - 最大化专家行为的恢复奖励
        if expert_indices is not None and len(expert_indices) > 0:
            expert_loss = -recovered_rewards[expert_indices].mean()
            
            # 6. 计算间隔损失 - 确保专家Q值高于非专家Q值
            if non_expert_indices is not None and len(non_expert_indices) > 0:
                expert_q = current_q[expert_indices]
                non_expert_q = current_q[non_expert_indices]
                
                # 正确方向: 专家Q值应当高于非专家Q值
                margin_loss = F.relu(non_expert_q.mean() - expert_q.mean() + 1.0)
        
        # 7. 贝尔曼损失 - 确保Q值满足贝尔曼方程
        bellman_loss = F.mse_loss(current_q, target_q.detach())
        
        # 8. 组合Critic损失
        critic_loss = expert_loss + margin_loss + 0.5 * bellman_loss
        
        # 9. Actor损失 - 最大化预期Q值和熵正则化
        # 对每个状态采样动作
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 计算每个动作的Q值期望
        q_values = torch.zeros_like(probs)
        for a in range(self.action_dim):
            actions = torch.full((state.shape[0],), a, device=self.device, dtype=torch.long)
            q_a = self.critic_fn(state, actions).squeeze()
            q_values[:, a] = q_a
        
        # 策略梯度: 最大化 E_a~π[Q(s,a)] + α·H(π)
        policy_loss = -(probs * q_values).sum(dim=1).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        actor_loss = policy_loss - self.alpha * entropy
        
        return critic_loss, actor_loss, {
            'expert_loss': expert_loss.item(),
            'margin_loss': margin_loss.item(),
            'bellman_loss': bellman_loss.item(),
            'entropy': entropy.item(),
            'recovered_rewards': recovered_rewards.mean().item()
        }
    
    def soft_update(self):
        """软更新目标网络参数"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic_sa.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
def validate_softq_agent():
    """验证Soft Q-Learning框架在简单环境中的表现"""
    # 创建环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2
    # 加载专家数据
    expert_data = load_expert_data()
    print(f"专家数据维度: states {expert_data[0].shape}, actions {expert_data[1].shape}")
    
    # SoftQ特有的参数设置
    args = Namespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        gamma=0.99,
        critic_tau=0.005,
        min_temperature=0.1,          # 最小温度参数
        temperature=1.0,              # 初始温度
        temperature_decay=0.995,      # 温度衰减系数
        batch_size=128
    )
    
    # 初始化简化版的SoftQ智能体
    class SimpleSoftQAgent(nn.Module):
        def __init__(self, state_dim, action_dim, args):
            super().__init__()
            self.device = args.device
            self.gamma = args.gamma
            self.tau = args.critic_tau
            self.total_updates = 0
            self.temperature = args.temperature
            self.min_temperature = args.min_temperature
            self.temperature_decay = args.temperature_decay
            self.action_dim = action_dim
            
            # 双Q网络
            self.q_network1 = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            )
            
            self.q_network2 = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            )
            
            # 目标网络
            self.target_q1 = copy.deepcopy(self.q_network1)
            self.target_q2 = copy.deepcopy(self.q_network2)
            for param in self.target_q1.parameters():
                param.requires_grad = False
            for param in self.target_q2.parameters():
                param.requires_grad = False
        
        def forward(self, state):
            """获取所有动作的Q值"""
            q1 = self.q_network1(state)
            q2 = self.q_network2(state)
            return torch.min(q1, q2)  # 直接比较两个张量的对应元素
        
        def get_action(self, state, exploration=True):
            """基于SoftQ策略选择动作"""
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.forward(state).squeeze()
                
                if exploration:
                    # 使用温度参数生成概率分布
                    self.temperature = max(self.min_temperature, 
                                            self.temperature * 0.998)  # 使用缓慢的线性衰减
                    probs = F.softmax(q_values / self.temperature, dim=-1)
                    return torch.multinomial(probs, 1).item()
                else:
                    return torch.argmax(q_values).item()
        
        def compute_v(self, state, target=False):
            """计算软最大值形式的状态值"""
            if target:
                q1 = self.target_q1(state)
                q2 = self.target_q2(state)
                q = torch.min(q1, q2)
            else:
                q1 = self.q_network1(state)
                q2 = self.q_network2(state)
                q = torch.min(q1, q2)
            
            return self.temperature * torch.logsumexp(q / self.temperature, dim=1, keepdim=True)
        
        def compute_loss(self, batch):
            """计算SoftQ损失"""
            states, next_states, actions, rewards, dones = batch
            
            # 1. 当前Q值
            q1_all = self.q_network1(states)
            q2_all = self.q_network2(states)
            q1 = q1_all.gather(1, actions)
            q2 = q2_all.gather(1, actions)
            
            # 2. 目标Q值
            with torch.no_grad():
                next_v = self.compute_v(next_states, target=True)
                target_q = rewards + self.gamma * (1 - dones) * next_v
            
            # 3. TD误差
            td_error1 = q1 - target_q
            td_error2 = q2 - target_q
            
            # 4. 损失函数
            loss1 = F.mse_loss(q1, target_q)
            loss2 = F.mse_loss(q2, target_q)
            
            return loss1 + loss2, torch.abs(td_error1 + td_error2) / 2
        
        def update_temperature(self):
            """更新温度参数"""
            self.temperature = max(self.min_temperature, 
                                  self.temperature * self.temperature_decay)
        def compute_softq_loss_with_expert(self, batch):
            """SoftQ损失计算，支持专家样本权重"""
            states, next_states, actions, rewards, dones, expert_mask, expert_weights = batch
            
            # 1. 当前Q值
            q1_all = self.q_network1(states)
            q2_all = self.q_network2(states)
            q1 = q1_all.gather(1, actions)
            q2 = q2_all.gather(1, actions)
            
            # 2. 目标Q值
            with torch.no_grad():
                next_v = self.compute_v(next_states, target=True)
                target_q = rewards + self.gamma * (1 - dones) * next_v
            
            # 3. TD误差
            td_error1 = q1 - target_q
            td_error2 = q2 - target_q
            
            # 4. 根据专家样本施加额外权重
            mse1 = F.mse_loss(q1, target_q, reduction='none')
            mse2 = F.mse_loss(q2, target_q, reduction='none')
            
            # 应用专家权重
            weighted_mse1 = (mse1 * expert_weights).mean()
            weighted_mse2 = (mse2 * expert_weights).mean()
            
            # 5. 专家行为特殊损失 - 让专家行为的Q值更高
            if expert_mask.any():
                # 提取专家行为的Q值
                expert_q1 = q1[expert_mask]
                expert_q2 = q2[expert_mask]
                
                # 提高专家行为的Q值
                expert_loss = -(expert_q1.mean() + expert_q2.mean()) * 0.5
                
                # 总损失
                loss = weighted_mse1 + weighted_mse2 + 0.1 * expert_loss
            else:
                loss = weighted_mse1 + weighted_mse2
            
            return loss, torch.abs(td_error1 + td_error2) / 2
        def soft_update(self):
            """软更新目标网络"""
            for target_param, param in zip(self.target_q1.parameters(), self.q_network1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
            for target_param, param in zip(self.target_q2.parameters(), self.q_network2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    # 实例化智能体并训练
    agent = SimpleSoftQAgent(state_dim, action_dim, args).to(args.device)
    optimizer = torch.optim.Adam(list(agent.q_network1.parameters()) + 
                                 list(agent.q_network2.parameters()), lr=3e-4)
    
    # 优先级缓冲区实现
    class PrioritizedReplayBuffer:
        def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
            """
            初始化优先级缓冲区
            参数:
                capacity: 缓冲区大小
                alpha: 优先级指数 (0表示均匀采样)
                beta: 初始重要性采样指数 (纠正IS偏差)
                beta_increment: beta随时间线性增加的量
            """
            self.capacity = capacity
            self.alpha = alpha
            self.beta = beta
            self.beta_increment = beta_increment
            self.buffer = []
            self.priorities = np.zeros(capacity, dtype=np.float32)
            self.position = 0
        
        def store(self, state, next_state, action, reward, done):
            """存储新的转换，分配最高优先级"""
            max_priority = self.priorities.max() if self.buffer else 1.0
            
            if len(self.buffer) < self.capacity:
                self.buffer.append((state, next_state, action, reward, done))
            else:
                self.buffer[self.position] = (state, next_state, action, reward, done)
            
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity
        def __len__(self):
            """返回缓冲区中当前样本数量"""
            return len(self.buffer)
        def sample(self, batch_size):
            """基于优先级进行采样，返回重要性采样权重"""
            if len(self.buffer) < batch_size:
                return None, None, None
            
            buffer_len = min(len(self.buffer), self.capacity)
            
            # 计算采样概率 (根据优先级)
            priorities = self.priorities[:buffer_len] ** self.alpha
            probs = priorities / priorities.sum()
            
            # 随机抽样
            indices = np.random.choice(buffer_len, batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            
            # 计算重要性采样权重
            weights = (buffer_len * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # 归一化权重
            
            # 增加beta以逐渐减少偏差
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            return samples, indices, torch.FloatTensor(weights).reshape(-1, 1)
        
        def update_priorities(self, indices, td_errors):
            """根据TD误差更新优先级"""
            for idx, error in zip(indices, td_errors):
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
    # 经验回放缓冲区
    buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta=0.4)
    
    # 训练参数
    n_updates = 500
    eval_interval = 10
    eval_episodes = 5
    rewards_history = []
    temp_history = []
    # 训练参数
    n_updates = 500
    eval_interval = 10
    expert_ratio = 0.5  # 每个批次中专家数据的比例
    
    print("开始SoftQ验证训练...")
    for update in range(n_updates):
        agent.total_updates = update
        
        # 评估
        if update % eval_interval == 0:
            total_rewards = []
            for _ in range(eval_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = agent.get_action(state, exploration=False)
                    next_state, reward, terminated, _, = env.step(action)
                    done = terminated 
                    episode_reward += reward
                    state = next_state
                total_rewards.append(episode_reward)
            mean_reward = np.mean(total_rewards)
            rewards_history.append(mean_reward)
            temp_history.append(agent.temperature)
            print(f"Update {update}, Mean Reward: {mean_reward:.2f}, Temperature: {agent.temperature:.4f}")
        
        # 收集数据
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = agent.get_action(state, exploration=True)
            next_state, reward, terminated, _= env.step(action)
            done = terminated 
            buffer.store(state, next_state, action, reward, done)
            state = next_state
            steps += 1
        
        # 训练批次
        # 训练批次
        if buffer.__len__() >= args.batch_size // 2:  # 确保有足够的在线数据
            # 1. 从专家数据中采样
            
            expert_indices = np.random.randint(0, len(expert_data[0]), int(args.batch_size * expert_ratio))
            expert_states = torch.FloatTensor(expert_data[0][expert_indices]).to(args.device)
            expert_actions = torch.LongTensor(expert_data[1][expert_indices]).to(args.device)
            expert_next_states = torch.FloatTensor(expert_data[2][expert_indices]).to(args.device)
            expert_rewards = torch.FloatTensor(expert_data[3][expert_indices]).reshape(-1, 1).to(args.device)
            expert_dones = torch.FloatTensor(expert_data[4][expert_indices]).reshape(-1, 1).to(args.device)
            # 使用优先级采样
            online_samples, online_indices, online_weights = buffer.sample(args.batch_size // 2)
            
            if online_samples:
                online_states = torch.FloatTensor([s[0] for s in online_samples]).to(args.device)
                online_next_states = torch.FloatTensor([s[1] for s in online_samples]).to(args.device)
                online_actions = torch.LongTensor([[s[2]] for s in online_samples]).to(args.device)
                online_rewards = torch.FloatTensor([[s[3]] for s in online_samples]).to(args.device)
                online_dones = torch.FloatTensor([[s[4]] for s in online_samples]).to(args.device)
                
                # 合并专家和在线数据
                states = torch.cat([expert_states, online_states], dim=0)
                next_states = torch.cat([expert_next_states, online_next_states], dim=0)
                actions = torch.cat([expert_actions.unsqueeze(1), online_actions], dim=0)
                rewards = torch.cat([expert_rewards, online_rewards], dim=0)
                dones = torch.cat([expert_dones, online_dones], dim=0)
                
                # 重要性权重 - 对专家数据始终使用1.0，对在线数据使用计算的权重
                batch_weights = torch.ones(len(states), 1).to(args.device)
                batch_weights[len(expert_states):] = online_weights
                
                # 专家掩码
                expert_mask = torch.zeros(len(states), dtype=torch.bool).to(args.device)
                expert_mask[:len(expert_states)] = True
                
                # 计算损失并获取TD误差
                loss, td_errors = agent.compute_softq_loss_with_expert(
                    (states, next_states, actions, rewards, dones, expert_mask, batch_weights))
                
                # 更新优先级
                if online_indices is not None:
                    td_errors_online = td_errors[len(expert_states):].detach().cpu().numpy()
                    buffer.update_priorities(online_indices, td_errors_online)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 软更新目标网络
            if update % 2 == 0:
                agent.soft_update()
            
            # 更新温度参数
            agent.update_temperature()
    
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, n_updates, eval_interval), rewards_history)
    plt.xlabel('Updates')
    plt.ylabel('Average Reward')
    plt.title('SoftQ Training Progress')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, n_updates, eval_interval), temp_history)
    plt.xlabel('Updates')
    plt.ylabel('Temperature')
    plt.title('Temperature Decay')
    
    plt.tight_layout()
    plt.savefig('softq_validation.png')
    
    print(f"SoftQ验证完成，最终平均奖励: {rewards_history[-1]:.2f}")
    return agent, rewards_history
# 专家数据收集函数
def inspect_expert_data(filepath="F:/MineRL/VPT/CartPole-v1_1000.npy"):
    """详细检查专家数据的结构"""
    try:
        data = np.load(filepath, allow_pickle=True).item()
        print("专家数据键:", data.keys())
        
        for key in data:
            shape_info = f"{key} 形状: {np.array(data[key]).shape}"
            print(shape_info)
            
        # 分析states数据结构
        if 'states' in data and len(data['states'].shape) == 3:
            print("\n状态数据分析:")
            print(f"第一个episode第一个状态: {data['states'][0, 0]}")
            print(f"第一个episode最后一个状态: {data['states'][0, -1]}")
            
            # 检查是否每个episode都是500步
            non_zero_steps = [np.sum(np.any(data['states'][i] != 0, axis=1)) for i in range(min(10, len(data['states'])))]
            print(f"前10个episode的非零状态步数: {non_zero_steps}")
        
        # 分析actions数据结构
        if 'actions' in data and len(data['actions'].shape) == 2:
            print("\n动作数据分析:")
            print(f"第一个episode的前10个动作: {data['actions'][0, :10]}")
            unique_actions = np.unique(data['actions'])
            print(f"所有不同的动作值: {unique_actions}")
        
        # 分析rewards数据结构
        if 'rewards' in data and len(data['rewards'].shape) == 2:
            print("\n奖励数据分析:")
            print(f"第一个episode的前10个奖励: {data['rewards'][0, :10]}")
            print(f"奖励范围: [{np.min(data['rewards'])}, {np.max(data['rewards'])}]")
            
            # 检查每个episode的总奖励
            episode_rewards = [np.sum(data['rewards'][i]) for i in range(min(10, len(data['rewards'])))]
            print(f"前10个episode的总奖励: {episode_rewards}")
        
        return data
    except Exception as e:
        print(f"分析专家数据失败: {e}")
        return None

# 在主函数中调用

def load_expert_data(filepath="F:/MineRL/VPT/CartPole-v1_1000.npy"):
    """加载专家数据并转换为适合强化学习的格式"""
    try:
        data = np.load(filepath, allow_pickle=True).item()
        print(f"加载了{len(data['states'])}个专家轨迹")
        
        # 使用convert_trajectory_data函数处理轨迹数据
        processed_data = convert_trajectory_data(data)
        
        print(f"处理后获得{len(processed_data[0])}个转换样本")
        print(f"处理后数据维度: states {processed_data[0].shape}, actions {processed_data[1].shape}")
        return processed_data
        
    except Exception as e:
        print(f"加载专家数据失败: {e}")
        return None

# 主验证函数
def validate_iql_agent():
    # 创建环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2
    
    # 参数设置 - 使用更合适的超参数
    args = Namespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        gamma=0.99,
        critic_tau=0.005,
        init_alpha=0.05,  # 降低熵正则系数
        batch_size=128    # 增大批量大小以提高学习稳定性
    )
    
    # 初始化智能体
    agent = SimpleIQLAgent(state_dim, action_dim, args).to(args.device)
    
    # 加载专家数据
    expert_data = load_expert_data()
    print(f"处理后专家数据维度: states {expert_data[0].shape}, actions {expert_data[1].shape}")
    
    # 收集在线数据
    online_buffer = deque(maxlen=10000)
    
    # 优化器
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(agent.critic_sa.parameters(), lr=3e-4)
    
    # 学习率调度器
    actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=200, gamma=0.5)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=200, gamma=0.5)
    
    # 训练参数
    n_updates = 1000
    eval_interval = 10
    eval_episodes = 5
    rewards_history = []
    metrics_history = {'expert_loss': [], 'margin_loss': [], 'bellman_loss': [], 'entropy': [], 'recovered_rewards': []}
    
    # BC预训练阶段
    
    # IQL训练循环
    print("开始IQ-Learn训练...")
    for update in range(n_updates):
        agent.total_updates = update
        
        # 每隔一段时间进行评估
        if update % eval_interval == 0:
            total_rewards = []
            for _ in range(eval_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = agent.get_action(state, exploration=False)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                total_rewards.append(episode_reward)
            mean_reward = np.mean(total_rewards)
            rewards_history.append(mean_reward)
            print(f"Update {update}, Mean Reward: {mean_reward:.2f}")
        
        # 收集在线数据
        if update % 5 == 0:  # 更频繁收集在线数据
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < 200:  # 限制每次收集的步数
                action = agent.get_action(state, exploration=True)
                next_state, reward, done, _ = env.step(action)
                online_buffer.append((state, action, next_state, reward, done))
                state = next_state
                steps += 1
                
        # 准备批次数据
        # 从专家数据中采样
        expert_indices = np.random.randint(0, len(expert_data[0]), args.batch_size // 2)
        expert_states = expert_data[0][expert_indices]
        expert_actions = expert_data[1][expert_indices]
        expert_next_states = expert_data[2][expert_indices]
        expert_rewards = expert_data[3][expert_indices]
        expert_dones = expert_data[4][expert_indices]
        
        # 确保维度正确
        if len(expert_rewards.shape) == 1:
            expert_rewards = expert_rewards.reshape(-1, 1)
        if len(expert_dones.shape) == 1:
            expert_dones = expert_dones.reshape(-1, 1)
        expert_is_expert = np.ones((args.batch_size // 2, 1))

        # 从在线数据中采样
        if len(online_buffer) >= args.batch_size // 2:
            online_batch = random.sample(online_buffer, args.batch_size // 2)
            online_states = np.array([x[0] for x in online_batch])
            online_actions = np.array([x[1] for x in online_batch])
            online_next_states = np.array([x[2] for x in online_batch])
            online_rewards = np.array([x[3] for x in online_batch]).reshape(-1, 1)
            online_dones = np.array([x[4] for x in online_batch]).reshape(-1, 1)
            online_is_expert = np.zeros((args.batch_size // 2, 1))
            
            # 合并专家和在线数据
            states = np.concatenate([expert_states, online_states])
            actions = np.concatenate([expert_actions, online_actions])
            next_states = np.concatenate([expert_next_states, online_next_states])
            rewards = np.concatenate([expert_rewards, online_rewards])
            dones = np.concatenate([expert_dones, online_dones])
            is_expert = np.concatenate([expert_is_expert, online_is_expert])
        else:
            # 如果没有足够的在线数据，就全部使用专家数据
            states = expert_states
            actions = expert_actions
            next_states = expert_next_states
            rewards = expert_rewards
            dones = expert_dones
            is_expert = expert_is_expert
        
        # 转换为张量
        states = torch.FloatTensor(states).to(args.device)
        actions = torch.LongTensor(actions).to(args.device)
        next_states = torch.FloatTensor(next_states).to(args.device)
        rewards = torch.FloatTensor(rewards).to(args.device)
        dones = torch.FloatTensor(dones).to(args.device)
        is_expert = torch.BoolTensor(is_expert).to(args.device)
        
        # 计算损失并更新
        batch = (states, next_states, actions, rewards, dones, is_expert)
        
        # 先更新Critic
        critic_loss, _, metrics = agent.compute_losses(batch)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_sa.parameters(), 1.0)  # 梯度裁剪
        critic_optimizer.step()
        
        # 再更新Actor
        _, actor_loss, _ = agent.compute_losses(batch)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)  # 梯度裁剪
        actor_optimizer.step()
        # 调用软更新目标网络
        agent.soft_update()

        # 更新熵系数
        agent.update_alpha()
        # 软更新目标网络
        if update % 2 == 0:  # 更频繁地更新目标网络
            agent.soft_update()
            
        # 更新学习率
        if update % 50 == 0:
            actor_scheduler.step()
            critic_scheduler.step()
            
        # 记录指标
        for k, v in metrics.items():
            if k in metrics_history:
                metrics_history[k].append(v)
    
    # 绘制训练结果
    plt.figure(figsize=(15, 10))
    
    # 奖励曲线
    plt.subplot(2, 3, 1)
    plt.plot(range(0, n_updates, eval_interval), rewards_history)
    plt.xlabel('Updates')
    plt.ylabel('Average Reward')
    plt.title('IQL Agent Training Progress')
    
    # 各种损失曲线
    metrics_to_plot = ['expert_loss', 'margin_loss', 'bellman_loss', 'entropy', 'recovered_rewards']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+2)
        plt.plot(metrics_history[metric])
        plt.xlabel('Updates')
        plt.ylabel(metric)
        plt.title(f'{metric} over training')
    
    plt.tight_layout()
    plt.savefig('iql_validation_detailed.png')
    
    return agent, rewards_history
def convert_trajectory_data(data):
    """将轨迹数据转换为RL训练所需的转换元组"""
    states_list = []
    next_states_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    
    n_episodes = data['states'].shape[0]
    max_steps = data['states'].shape[1]
    
    for ep in range(n_episodes):
        # 找出这个episode的实际步数（非零状态）
        valid_mask = np.any(data['states'][ep] != 0, axis=1)
        valid_steps = np.sum(valid_mask)
        
        if valid_steps <= 1:  # 至少需要两个状态才能构成一个转换
            continue
            
        for t in range(valid_steps-1):
            states_list.append(data['states'][ep, t])
            next_states_list.append(data['states'][ep, t+1])
            actions_list.append(data['actions'][ep, t])
            rewards_list.append(data['rewards'][ep, t])
            # 只有最后一步是done=True
            dones_list.append(1.0 if t == valid_steps-2 else 0.0)
    
    return (
        np.array(states_list),
        np.array(actions_list),
        np.array(next_states_list),
        np.array(rewards_list),
        np.array(dones_list)
    )
if __name__ == "__main__":
    #data_analysis = inspect_expert_data()
   # agent, rewards = validate_iql_agent()
   # print(f"Final performance: {rewards[-1]:.2f}")
    # 然后运行验证
    torch.autograd.set_detect_anomaly(True)
    agent, rewards = validate_iql_agent()
    print(f"Final performance: {rewards[-1]:.2f}")