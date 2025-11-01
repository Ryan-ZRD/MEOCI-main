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
# Adaptive Dual-Pool D3QN Agent (Algorithm 1)  — UPDATED
# -------------------------------------------------------------------------
class ADPD3QNAgent:
    """
    ADP-D3QN智能体（Algorithm 1）
    - 兼容四种变体：
      D3QN            : use_adaptive=False, use_dual_pool=False
      A-D3QN          : use_adaptive=True,  use_dual_pool=False
      DP-D3QN         : use_adaptive=False, use_dual_pool=True
      ADP-D3QN        : use_adaptive=True,  use_dual_pool=True
    - 兼容 training.py 期望的接口：store_experience / learn / update_epsilon / E1 E2
    """

    def __init__(self, state_dim, action_dim, config,
                 use_adaptive=False, use_dual_pool=False):
        # ======= 配置对齐 =======
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = config.get("gamma", 0.9)
        self.lr = config.get("lr", 0.01)
        # training.py 里用的是 config.batch_size
        self.batch_size = config.get("batch_size", config.get("drl_batch_size", 128))
        # epsilon 命名对齐
        self.epsilon_max = config.get("epsilon_max", config.get("eps_max", 1.0))
        self.epsilon_min = config.get("epsilon_min", config.get("eps_min", 0.05))
        self.epsilon_decay = config.get("epsilon_decay", config.get("eps_decay", 0.995))
        # 经验池容量命名对齐
        self.buffer_size = config.get("buffer_size", config.get("exp_pool_size", 3000))
        # 采样比例
        self.lambda_high = config.get("lambda_high", config.get("sample_ratio", 0.7))
        self.lambda_low = 1.0 - self.lambda_high
        # 目标网更新频率
        self.update_freq = config.get("update_freq", config.get("target_update_freq", 10))
        self.hidden_dim = config.get("hidden_dim", 128)
        self.action_dim = action_dim
        self.state_dim = state_dim

        # 变体开关
        self.use_adaptive = use_adaptive
        self.use_dual_pool = use_dual_pool

        # === 经验池 ===
        if self.use_dual_pool:
            self.memory_high = deque(maxlen=self.buffer_size)  # E1
            self.memory_low = deque(maxlen=self.buffer_size)   # E2
        else:
            self.replay_buffer = deque(maxlen=self.buffer_size)  # 单池

        # === Q网络 ===
        self.q_net = DuelingQNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.epsilon = self.epsilon_max
        self.learn_step_counter = 0

    # ---------- 兼容 training.py 调用的属性名 ----------
    @property
    def E1(self):
        return self.memory_high if self.use_dual_pool else self.replay_buffer

    @property
    def E2(self):
        return self.memory_low if self.use_dual_pool else deque()  # 空占位

    # ---------------------------------------------------------------------
    # ε-greedy 动作选择 (Eq.18)
    # ---------------------------------------------------------------------
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state)
        return torch.argmax(q_values, dim=1).item()

    # ---------------------------------------------------------------------
    # 经验存储（兼容 training.py 的 store_experience）
    # ---------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        """内部真实实现：按Q值正负粗分高低池"""
        if self.use_dual_pool:
            with torch.no_grad():
                qv = self.q_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))[0, action].item()
            if qv > 0:
                self.memory_high.append((state, action, reward, next_state, done))
            else:
                self.memory_low.append((state, action, reward, next_state, done))
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))

    # >>> 新增：training.py 期望的函数名
    def store_experience(self, state, action, reward, next_state, done):
        self.store_transition(state, action, reward, next_state, done)

    # ---------------------------------------------------------------------
    # 更新（兼容 training.py 的 learn()，返回 loss 值）
    # ---------------------------------------------------------------------
    def update(self):
        if self.use_dual_pool:
            if len(self.memory_high) + len(self.memory_low) < self.batch_size:
                return 0.0
            n_high = int(self.batch_size * self.lambda_high)
            n_low = self.batch_size - n_high
            batch_high = random.sample(self.memory_high, min(n_high, len(self.memory_high)))
            batch_low = random.sample(self.memory_low, min(n_low, len(self.memory_low)))
            batch = batch_high + batch_low
        else:
            if len(self.replay_buffer) < self.batch_size:
                return 0.0
            batch = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states), dim=1, keepdim=True)
            q_targets_next = self.target_net(next_states).gather(1, next_actions)
            q_targets = rewards + (1 - dones) * self.gamma * q_targets_next

        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # epsilon 调整：自适应 vs 固定衰减
        if self.use_adaptive:
            # 例：按近期TD误差或成功率动态也可，这里保守用更快衰减
            self.epsilon = max(self.epsilon * (self.epsilon_decay ** 1.2), self.epsilon_min)
        else:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return float(loss.item())

    # >>> 新增：training.py 期望的 learn() 名称
    def learn(self):
        return self.update()

    # ---------------------------------------------------------------------
    # epsilon 更新（兼容 training.py 的 update_epsilon(epoch, max_epoch)）
    # ---------------------------------------------------------------------
    def update_epsilon(self, epoch, epoch_max):
        # 线性调度（与自适应并存，二者取较小值）
        lin_eps = self.epsilon_max + (self.epsilon_min - self.epsilon_max) * (epoch / max(1, epoch_max))
        self.epsilon = max(min(self.epsilon, lin_eps), self.epsilon_min)

    # 保存 / 加载（保持不变）
    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
