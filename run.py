import os
import torch
import gym
import imageio  # 用于保存视频
from IQLAgent import IQLearnAgent  # 独立训练的IQ-Learn模型
import minerl
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from argparse import ArgumentParser
import torchvision.transforms as T
import imageio  # 新增导入 imageio，用于保存视频
# 创建环境
env = gym.make('MineRLBasaltFindCave-v0')
RESOLUTION = (72, 128)
# 初始化 IGL 模型（此处假设 IQLearnAgent 就是你想调用的 igl 模型）
agent = IQLearnAgent().to('cuda')  # 将模型迁移到 GPU
agent.load('F:\MineRL\VPT\model_save\iql_agent.pth')  # 替换为模型检查点路径

frames = []  # 用于保存渲染帧
num_episodes = 10  # 调整探索的总回合数
class CaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            test_out = self.cnn(torch.zeros(1,3,*RESOLUTION)).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(test_out, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return F.softmax(self.classifier(self.cnn(x)), dim=1)
cave=CaveDiscriminator()

if os.path.exists('cave_detector_best.pth'):
    cave_state = torch.load('cave_detector_best.pth')['model_state_dict']
    cave.load_state_dict(cave_state)
    print("Loaded Cave Discriminator")
cave = cave.to('cuda')  # 确保模型在 GPU 上
agent.device='cuda'
for ep in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        trans_pipeline = T.Compose([
            T.ToPILImage(),
            T.Resize((72,128)),
            T.ToTensor()
        ])
        # 将 obs['pov'] 转换后的张量移动到 GPU
        # 将 obs['pov'] 转换后的张量移动到 GPU，注意 unsqueeze 添加 batch 维用于 cave 模型
        pov_tensor = trans_pipeline(obs['pov']).unsqueeze(0).to('cuda')
        cave_prob = cave(pov_tensor)[:, 1].unsqueeze(1)
        # 移除 batch 维度，将图像移回 CPU，以便 agent.transform 正常工作
        obs['pov'] = pov_tensor.squeeze(0).cpu()

        action = agent.get_action(obs, cave_prob)
        action["ESC"] = 0
        next_obs, reward, done, _ = env.step(action)
        preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((128,128)),
        T.ToTensor()
        ])
        processed_obs = preprocess(obs["pov"]).cpu().numpy()
        processed_next_obs = preprocess(next_obs["pov"]).cpu().numpy()
        diff_metric = np.mean(np.abs(processed_obs.astype(np.float32) - processed_next_obs.astype(np.float32)))
        #print(diff_metric)
        print(cave_prob)
        env.render()
# 保存视频到当前工作目录
'''
video_path = os.path.join(os.getcwd(), "exploration_video.mp4")
imageio.mimsave(video_path, frames, fps=30)
print(f"视频已保存至：{video_path}")
'''