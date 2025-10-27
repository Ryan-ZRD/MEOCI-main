import torch
import numpy as np
import random
import os

class Config:
    """
    MEOCI: Model Partitioning and Early-exit point Selection Joint Optimization
    for Collaborative Inference in Vehicular Edge Computing
    ------------------------------------------------------------
    全局实验配置文件，对应论文 Table 1–147 与 Section 3–5 参数设置。
    """

    def __init__(self, model_name="MultiExitVGG16"):
        # ==========================================================
        # 1. 实验随机性控制（确保复现性）
        # ==========================================================
        self.random_seed = 42
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # ==========================================================
        # 2. 模型特定参数映射（统一模型配置表）
        # ==========================================================
        MODEL_CONFIGS = {
            "MultiExitAlexNet": {
                "num_layers": 12,
                "num_classes": 10,
                "delay_tolerate": 25.0,   # ms
                "action_dim": 13,
                "exit_count": 4
            },
            "MultiExitVGG16": {
                "num_layers": 31,
                "num_classes": 10,
                "delay_tolerate": 250.0,  # ms
                "action_dim": 32,
                "exit_count": 5
            },
            "MultiExitResNet50": {
                "num_layers": 49,
                "num_classes": 10,
                "delay_tolerate": 300.0,
                "action_dim": 50,
                "exit_count": 6
            },
            "MultiExitYOLOv10": {
                "num_layers": 70,
                "num_classes": 7,         # Detection categories
                "delay_tolerate": 400.0,
                "action_dim": 71,
                "exit_count": 3
            }
        }

        assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"
        model_cfg = MODEL_CONFIGS[model_name]
        self.model_name = model_name

        # ==========================================================
        # 3. 数据配置（Section 5.1: Dataset & Preprocessing）
        # ==========================================================
        self.data_root = "/path/to/bdd100k"
        self.img_size = (224, 224)
        self.train_batch_size = 32
        self.num_workers = 4
        self.augment = True
        self.num_classes = model_cfg["num_classes"]

        # ==========================================================
        # 4. 模型与任务配置（Section 3.2–3.4）
        # ==========================================================
        self.ac_min = 0.8                    # 最小推理精度 (Eq.15)
        self.initial_partition_point = 5     # 初始划分点 0<par<l
        self.initial_exit_idx = -1           # 主出口
        self.exit_count = model_cfg["exit_count"]
        self.num_layers = model_cfg["num_layers"]

        # ==========================================================
        # 5. 车边计算与通信参数（Table 1–147）
        # ==========================================================
        self.vehicle_device = "cpu"
        self.edge_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vehicle_resource = 1.5          # GHz
        self.edge_resource = 15.0            # GHz
        self.bandwidth = 20.0                # Mbps
        self.max_vehicle_power = 3.0         # W
        self.edge_power = 40.0               # dBm
        self.edge_queue_capacity = 50

        # ==========================================================
        # 6. 性能约束与代价模型（Eq.15d–15h）
        # ==========================================================
        self.delay_tolerate = model_cfg["delay_tolerate"]
        self.energy_tolerate = 25.0          # J
        self.per_layer_compute = 0.1         # GOPS/layer (approx)

        # ==========================================================
        # 7. ADP-D3QN算法配置（Section 4.3）
        # ==========================================================
        self.state_dim = 4                   # (ac, queue_e, c_e^rm, λ)
        self.action_dim = model_cfg["action_dim"]
        self.gamma = 0.9
        self.lr = 0.01
        self.drl_batch_size = 128
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 10
        self.exp_pool_size = 3000
        self.sample_ratio = 0.7
        self.max_epoch = 1000
        self.max_step_per_epoch = 200

        # ==========================================================
        # 8. 训练与日志配置
        # ==========================================================
        self.save_root = "./saved_models"
        self.save_interval = 100
        self.log_interval = 10

        # ==========================================================
        # 9. 输出确认
        # ==========================================================
        print(f"[Config Loaded] Model: {self.model_name} | Delay Tolerance: {self.delay_tolerate} ms | "
              f"Action Dim: {self.action_dim} | Classes: {self.num_classes}")
        print(f"[Device] Vehicle: {self.vehicle_device} | Edge: {self.edge_device}")
        print(f"[DRL] γ={self.gamma}, lr={self.lr}, batch={self.drl_batch_size}")
