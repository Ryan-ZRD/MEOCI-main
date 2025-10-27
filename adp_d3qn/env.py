import numpy as np
import torch
from typing import Tuple, Dict, Any

from model.base_model import BaseMultiExitModel
from adp_d3qn.metrics import calculate_avg_delay, calculate_task_completion_rate


class VECEnv:
    """
    Vehicle–Edge Collaborative Environment
    --------------------------------------
    对应论文 Section IV-B 的 MDP 定义 S,A,P,R,γ
    - 状态 s(t) = (ac(t), queue_e(t), c_e^rm(t), λ(t))
    - 动作 a(t) = (par(t), exit(t))  其中 par∈{0..l}, exit∈{-1..n_exits-1}
      这里将 (par, exit) 映射为一个“扁平离散动作”，维度随模型动态变化：
        action_dim = (l + 1) * (n_exits + 1)
    - 转移包含：车端/边缘推理、M/D/1 排队、通信
    """

    def __init__(self, model: BaseMultiExitModel, dataloader, config: Dict[str, Any]):
        """
        :param model: 多出口 DNN 模型（AlexNet/VGG16/…）
            需提供：
              - model.backbone: list-like, 主干层集合（用于计算 l）
              - model.exit_layers: list-like, 各出口分类头（用于计算 n_exits）
              - model.partition_model(k): -> (vehicle_model, edge_model)
        :param dataloader: BDD100K 的数据加载器（返回 dict: {"image","label"}）
        :param config: 实验参数（全部可在 config.py 中改）
        """
        self.model = model
        self.dataloader = dataloader
        self.cfg = config
        self.device = torch.device(self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # === 动态动作空间 ===
        self.num_layers = int(len(getattr(self.model, "backbone")))
        self.num_exits = int(len(getattr(self.model, "exit_layers", [])))  # 可能为 0
        # par(t)∈[0..l]；exit∈{-1..num_exits-1}，我们把 "-1(不早退)" 平移为索引0
        self.action_dim = (self.num_layers + 1) * (self.num_exits + 1)

        # === 状态空间（固定4维，但归一化阈值可配） ===
        self.state_dim = 4

        # === 论文关键参数（全部可配，不写死） ===
        # 车辆数、资源、带宽、阈值等（对应 Table III）
        self.num_vehicles = self.cfg.get("num_vehicles", 10)
        self.edge_resource = float(self.cfg.get("edge_resource", 15.0))         # Ce (GHz)
        self.vehicle_resource = float(self.cfg.get("vehicle_resource", 1.5))    # Ci (GHz)
        self.bandwidth_mbps = float(self.cfg.get("bandwidth", 20.0))            # Brv (Mbps)
        self.delay_tolerate = float(self.cfg.get("delay_tolerate", 25.0))       # (ms)
        self.ac_min = float(self.cfg.get("ac_min", 0.8))                        # ac_min ∈ (0,0.8]
        self.gamma = float(self.cfg.get("gamma", 0.9))

        # 每层计算量（GOPS），若未配置则回退 0.1
        self.per_layer_compute = float(self.cfg.get("per_layer_compute", 0.1))

        # 队列归一化上限、到达率范围
        self.queue_norm_cap = int(self.cfg.get("queue_norm_cap", 50))
        self.arrival_rate_range = tuple(self.cfg.get("arrival_rate_range", (0.15, 0.25)))  # [15%,25%]
        self.arrival_rate_sigma = float(self.cfg.get("arrival_rate_sigma", 0.01))

        # 过载惩罚（用于防止 service_rate<=arrival_rate 的数值爆炸）
        self.edge_overload_penalty = float(self.cfg.get("edge_overload_penalty", 1e6))

        # 奖励权重（可在 config 中调参对齐论文消融）
        self.reward_alpha = float(self.cfg.get("reward_alpha", 0.5))  # 完成率
        self.reward_beta = float(self.cfg.get("reward_beta", 0.3))    # 精度

        self.reset()

    # ------------------------ 公共接口 ------------------------

    def reset(self) -> torch.Tensor:
        """重置环境状态 —— 对应 Section IV-B 的状态定义"""
        self.current_ac = float(np.random.uniform(self.ac_min, 1.0))         # ac(t)
        self.edge_queue = 0                                                  # queue_e(t)
        self.edge_remaining_resource = float(self.edge_resource)             # c_e^rm(t)
        low, high = self.arrival_rate_range
        self.task_arrival_rate = float(np.random.uniform(low, high))         # λ(t)

        self.data_iter = iter(self.dataloader)
        return self._get_state()

    def step(self, action: int):
        """
        环境一步交互 —— 对应 Section IV-B 的转移
        :param action: 扁平离散动作索引 ∈ [0, action_dim-1]
        :return: (next_state, reward, done, info)
        """
        par, exit_idx = self._decode_action(action)
        done = False

        # 1) 模型划分（Section III-B）
        vehicle_model, edge_model = self.model.partition_model(par)

        # 2) 取一批数据（BDD100K）
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        imgs = batch["image"].to(self.device, non_blocking=True)
        labels = batch.get("label", None)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        # 3) 推理与早退（Section III-C）
        with torch.no_grad():
            vehicle_feat = vehicle_model(imgs)

            if exit_idx >= 0 and exit_idx < self.num_exits:
                # 早退：在车端分支退出（对应 Eq.(1) 的早退概率逻辑的实现位点）
                exit_probs = self.model.exit_layers[exit_idx](vehicle_feat)          # (B, C)
                exit_probs = torch.softmax(exit_probs, dim=1)
                self.current_ac = float(torch.mean(exit_probs.max(dim=1).values).item())
                vehicle_delay = self._vehicle_delay(par)                             # Eq.(9)/(10) 期望近似
                edge_delay = 0.0
                transmission_delay = 0.0                                             # Eq.(6) 无传输
            else:
                # 不早退：传输中间特征到边缘 + 边缘主出口预测
                transmission_delay = self._transmission_delay(vehicle_feat)          # Eq.(5)/(6)
                edge_feat = edge_model(vehicle_feat)
                main_probs = torch.softmax(edge_feat, dim=1)
                self.current_ac = float(torch.mean(main_probs.max(dim=1).values).item())
                vehicle_delay = self._vehicle_delay(par)
                edge_delay = self._edge_delay(par)                                   # Eq.(8) M/D/1

        # 4) 核心指标（Section V-C）
        total_delay = vehicle_delay + transmission_delay + edge_delay
        avg_delay = calculate_avg_delay(total_delay, self.num_vehicles)              # Eq.(14)
        task_completion_rate = calculate_task_completion_rate(avg_delay, self.delay_tolerate)

        # 5) 状态转移（队列/资源/到达率演化）
        self._evolve_system(par)

        # 6) 奖励（R(t) = -delay_avg + α·completion + β·ac）
        reward = self._reward(avg_delay, task_completion_rate, self.current_ac)

        # 7) 终止条件（精度/资源）
        if self.current_ac < self.ac_min or self.edge_remaining_resource <= 0:
            done = True

        info = {
            "avg_delay": float(avg_delay),
            "task_completion_rate": float(task_completion_rate),
            "current_ac": float(self.current_ac),
            "vehicle_delay": float(vehicle_delay),
            "edge_delay": float(edge_delay),
            "transmission_delay": float(transmission_delay),
            "par": int(par),
            "exit_idx": int(exit_idx)
        }
        return self._get_state(), float(reward), bool(done), info

    # ------------------------ 内部函数 ------------------------

    def _decode_action(self, a: int) -> Tuple[int, int]:
        """
        将扁平离散动作索引 a 还原为 (par, exit)
        - par ∈ [0..l]
        - exit ∈ {-1..num_exits-1}   # 其中 -1 表示“不早退”
        """
        exits_plus = self.num_exits + 1
        par = a // exits_plus
        exit_flat = a % exits_plus
        exit_idx = exit_flat - 1      # 将 [0..num_exits] 平移到 [-1..num_exits-1]
        # 边界保护
        par = int(np.clip(par, 0, self.num_layers))
        exit_idx = int(np.clip(exit_idx, -1, self.num_exits - 1))
        return par, exit_idx

    def _get_state(self) -> torch.Tensor:
        """归一化状态向量 s(t)=(ac, queue, c_e^rm, λ)"""
        q_cap = max(1, self.queue_norm_cap)
        low, high = self.arrival_rate_range
        # 归一化
        state = np.array([
            np.clip(self.current_ac, 0.0, 1.0),                          # ac ∈ [0,1]
            np.clip(self.edge_queue / q_cap, 0.0, 1.0),                  # queue
            np.clip(self.edge_remaining_resource / self.edge_resource, 0.0, 1.0),
            np.clip((self.task_arrival_rate - low) / (high - low + 1e-9), 0.0, 1.0)
        ], dtype=np.float32)
        return torch.from_numpy(state)

    def _reward(self, avg_delay: float, completion_rate: float, current_ac: float) -> float:
        """
        奖励函数：
          R(t) = - delay_avg / delay_tolerate + α·completion_rate + β·ac
        """
        delay_penalty = - avg_delay / (self.delay_tolerate + 1e-9)
        return delay_penalty + self.reward_alpha * completion_rate + self.reward_beta * current_ac

    def _evolve_system(self, par: int) -> None:
        """队列长度、剩余资源、到达率的演化（近似模拟）"""
        # 队列：本轮到达任务 - 已服务 1 个（近似）
        arrivals = int(max(0, round(self.num_vehicles * self.task_arrival_rate)))
        self.edge_queue = max(0, self.edge_queue + arrivals - 1)

        # 剩余资源：用已在边缘端执行的层数占比粗略扣减
        edge_layers = max(0, self.num_layers - par)
        use_ratio = edge_layers / max(1, self.num_layers)
        self.edge_remaining_resource = max(0.0, self.edge_remaining_resource - use_ratio * 0.1 * self.edge_resource)

        # 到达率：加入噪声并裁剪在合法范围
        low, high = self.arrival_rate_range
        self.task_arrival_rate = float(np.clip(
            self.task_arrival_rate + np.random.normal(0.0, self.arrival_rate_sigma),
            low, high
        ))

    # ----- 各项延迟近似（对应论文方程处给出对齐注释） -----

    def _vehicle_delay(self, par: int) -> float:
        """
        车端处理延迟（近似 Eq.(9)/(10)）
          delay_icv(t) ≈ (∑_{j=1..par} c_j) / Ci
        这里用每层恒定计算量 per_layer_compute 近似 ∑ c_j
        返回单位：ms
        """
        if par <= 0:
            return 0.0
        compute_load = par * self.per_layer_compute        # GOPS
        delay_s = compute_load / max(self.vehicle_resource, 1e-9)  # s
        return float(delay_s * 1000.0)

    def _edge_delay(self, par: int) -> float:
        """
        边缘端排队+处理延迟（Eq.(8)：M/D/1）
          delay_oper(t) = 1/μ + λ / [2μ(μ-λ)]
          其中 μ = Ce / (∑_{j=par..l} c_j)
               λ ≈ 车辆任务到达率 * 车辆数（单位：任务/时间）
        为避免极端数值不稳，当 μ<=λ 时返回一个大惩罚。
        返回单位：ms
        """
        remaining_layers = max(0, self.num_layers - par)
        if remaining_layers == 0:
            return 0.0

        compute_load = remaining_layers * self.per_layer_compute         # GOPS
        mu = self.edge_resource / max(compute_load, 1e-9)                # “任务/秒”的近似
        lam = self.task_arrival_rate * self.num_vehicles                  # “任务/秒”的近似

        if mu <= lam:   # 过载：给极大延迟（或你可选择截断/软惩罚）
            return float(self.edge_overload_penalty)

        process = 1.0 / mu
        queue = lam / (2.0 * mu * (mu - lam) + 1e-9)
        return float((process + queue) * 1000.0)

    def _transmission_delay(self, feat: torch.Tensor) -> float:
        """
        通信/传输延迟（Eq.(5)/(6)）
          delay_trans = data_size(bits) / bandwidth(bits/s)
        用中间特征张量大小估算传输数据量（float32=4字节）
        返回单位：ms
        """
        with torch.no_grad():
            num_elems = int(feat.numel())
        data_mb = (num_elems * 4) / (1024.0 * 1024.0)     # MB
        bw_mbps = max(self.bandwidth_mbps, 1e-6)          # Mbps
        delay_s = (data_mb * 8.0) / bw_mbps               # 秒
        return float(delay_s * 1000.0)
