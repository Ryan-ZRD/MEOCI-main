import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DuelingQNetwork(nn.Module):
    """
    双分支Q网络（论文4.3节ADP-D3QN基础结构：价值函数V(s) + 优势函数A(s,a)）
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 价值函数分支（V(s)：状态价值）
        self.value_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 优势函数分支（A(s,a)：动作优势）
        self.advantage_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        """
        前向传播（论文4.3节：Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))）
        :param x: 状态（(B, state_dim)）
        :return: Q值（(B, action_dim)）
        """
        feat = self.feature_extractor(x)
        value = self.value_branch(feat)
        advantage = self.advantage_branch(feat)
        # 避免动作优势偏差，减去均值
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value

class ADPD3QNAgent:
    """
    ADP-D3QN智能体（论文4.3节核心算法：双经验池+自适应ε-greedy）
    """
    def __init__(self, state_dim, action_dim, config):
        """
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param config: 配置字典（论文1-147表格参数）
        """
        self.device = config["device"]
        self.config = config

        # 论文算法参数（1-147表格）
        self.gamma = config["gamma"]  # 折扣因子（0.9）
        self.lr = config["lr"]  # 学习率（0.01）
        self.batch_size = config["batch_size"]  # 批次