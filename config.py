import torch

class Config:
    def __init__(self):
        """
        论文MEOCI核心配置（严格遵循1-147表格参数）
        """
        # 1. 数据配置（论文5.1节）
        self.data_root = "/path/to/bdd100k"  # BDD100K数据集路径
        self.num_classes = 10  # 论文分类任务类别数
        self.img_size = (224, 224)  # 图像尺寸（AlexNet/VGG16输入）
        self.batch_size = 32  # 批次大小
        self.augment = True  # 训练时数据增强

        # 2. 模型配置（论文3.2-3.4节）
        self.model_name = "MultiExitVGG16"  # 可选：MultiExitAlexNet/MultiExitVGG16
        self.ac_min = 0.8  # 最小推理精度阈值（论文Eq.15）
        self.initial_partition_point = 5  # 初始划分点（0<par<l）
        self.initial_exit_idx = -1  # 初始早退出点（-1为主出口）

        # 3. 车边设备配置（论文5.1节）
        self.vehicle_device = "cpu"  # 车端设备（Raspberry Pi 4B/Jetson Nano）
        self.edge_device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 边缘端GPU
        self.vehicle_resource = 1.5  # 车端计算资源（1.5GHz，论文1-147）
        self.edge_resource = 15.0  # 边缘计算资源（15GHz，论文1-147）
        self.bandwidth = 20.0  # 通信带宽（20Mbps，论文1-147）
        self.max_vehicle_power = 3.0  # 车端最大功耗（3W，论文1-147）
        self.edge_power = 40.0  # 边缘端功耗（40dBm，论文1-147）

        # 4. 延迟与能量约束（论文Eq.15d-15h）
        self.delay_tolerate = 25.0 if self.model_name == "MultiExitAlexNet" else 250.0  # 容忍延迟（论文1-147）
        self.energy_tolerate = 25.0  # 容忍能量消耗（25J，论文1-147）
        self.edge_queue_capacity = 50  # 边缘队列容量

        # 5. ADP-D3QN算法配置（论文4.3节）
        self.state_dim = 4  # 状态维度：ac(t), queue_e(t), c_e^rm(t), λ(t)
        self.action_dim = 32 if self.model_name == "MultiExitVGG16" else 13  # 动作数=划分点数量（VGG1631层+1=32）
        self.gamma = 0.9  # 折扣因子（论文1-147）
        self.lr = 0.01  # 学习率（论文1-147）
        self.batch_size = 128  # 经验采样批次（论文1-147）
        self.epsilon_max = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.target_update_freq = 10  # 目标网络更新频率（论文1-147）
        self.exp_pool_size = 3000  # 经验池大小（E1=E2=1500，论文1-147）
        self.sample_ratio = 0.7  # 高价值经验采样比例（λ1=0.7，论文4.3节）
        self.max_epoch = 1000  # 最大训练轮次
        self.max_step_per_epoch = 200  # 每轮最大步数

        # 6. 训练与保存配置
        self.save_root = "./saved_models"  # 模型保存根路径
        self.save_interval = 100  # 每100轮保存一次模型
        self.log_interval = 10  # 每10步打印日志