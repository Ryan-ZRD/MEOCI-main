import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# -------------------------------------------------------------------------
# Dueling D3QN Network (Eq. for Q(s,a) = V(s) + (A(s,a) - mean(A)))
# -------------------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    """双分支Q网络（价值函数V(s) + 优势函数A(s,a)）"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        value = self.value_branch(feat)
        advantage = self.advantage_branch(feat)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value


# -------------------------------------------------------------------------
# Adaptive Dual-Pool D3QN Agent (Algorithm 1)
# -------------------------------------------------------------------------
class ADPD3QNAgent:
    """
    ADP-D3QN智能体（论文Algorithm 1核心）
    Step 1–14 实现：ε-greedy策略 + 双经验池 + Q网络更新
    """

    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = config.get("gamma", 0.9)
        self.lr = config.get("lr", 0.01)
        self.batch_size = config.get("batch_size", 128)
        self.epsilon_max = config.get("epsilon_max", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.buffer_size = config.get("buffer_size", 3000)
        self.lambda_high = config.get("lambda_high", 0.7)  # 高Q样本采样比例 λ1
        self.lambda_low = 1.0 - self.lambda_high            # 低Q样本采样比例 λ2
        self.update_freq = config.get("update_freq", 10)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.action_dim = action_dim
        self.state_dim = state_dim

        # === Step 1: 初始化两个经验池 E1 / E2 ===
        self.memory_high = deque(maxlen=self.buffer_size)
        self.memory_low = deque(maxlen=self.buffer_size)

        # === Step 2: 初始化主网络和目标网络 ===
        self.q_net = DuelingQNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.epsilon = self.epsilon_max
        self.learn_step_counter = 0

    # ---------------------------------------------------------------------
    # Step 3–4: ε-greedy 动作选择 (Eq. 18)
    # ---------------------------------------------------------------------
    def select_action(self, state):
        """
        动作选择策略 (论文Eq.18)
        ε = ε_max + (ε_min - ε_max) * (epoch / epoch_max)
        """
        if random.random() < self.epsilon:
            # 随机探索
            action = random.randint(0, self.action_dim - 1)
        else:
            # 贪婪选择最优动作
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            action = torch.argmax(q_values, dim=1).item()
        return action

    # ---------------------------------------------------------------------
    # Step 5: 存储经验到双经验池
    # ---------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        """根据Q值高低分配到不同经验池"""
        # 临时估计Q值判断优劣（高Q样本更优）
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_value = self.q_net(state_t)[0, action].item()
        if q_value > 0:  # 简单划分阈值，可根据经验动态调整
            self.memory_high.append((state, action, reward, next_state, done))
        else:
            self.memory_low.append((state, action, reward, next_state, done))

    # ---------------------------------------------------------------------
    # Step 6–10: Q网络更新 (mini-batch from E1 & E2)
    # ---------------------------------------------------------------------
    def update(self):
        """训练更新网络参数"""
        if len(self.memory_high) + len(self.memory_low) < self.batch_size:
            return

        # 混合采样高低Q经验
        n_high = int(self.batch_size * self.lambda_high)
        n_low = self.batch_size - n_high
        batch_high = random.sample(self.memory_high, min(n_high, len(self.memory_high)))
        batch_low = random.sample(self.memory_low, min(n_low, len(self.memory_low)))
        batch = batch_high + batch_low

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算当前Q(s,a)
        q_values = self.q_net(states).gather(1, actions)
        # 计算下一个状态的目标Q值
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states), dim=1, keepdim=True)
            q_targets_next = self.target_net(next_states).gather(1, next_actions)
            q_targets = rewards + (1 - dones) * self.gamma * q_targets_next

        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # === Step 11: 每Δ步更新目标网络 ===
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # === Step 12: 动态调整ε ===
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # ---------------------------------------------------------------------
    # Step 13–14: 保存与加载模型（论文训练稳定性验证）
    # ---------------------------------------------------------------------
    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
