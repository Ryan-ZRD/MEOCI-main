import numpy as np
import torch
from model.base_model import BaseMultiExitModel
from adp_d3qn.metrics import calculate_avg_delay, calculate_task_completion_rate

class VECEnv:
    """
    车边协同环境（论文4.2节MDP建模：S,A,P,R,γ）
    模拟VEC场景中任务到达、资源约束、延迟计算等核心逻辑
    """
    def __init__(self, model: BaseMultiExitModel, dataloader, config):
        """
        :param model: 多出口DNN模型（AlexNet/VGG16）
        :param dataloader: 数据加载器（BDD100K）
        :param config: 配置字典（论文1-147表格参数）
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config

        # 论文核心参数（1-147表格）
        self.num_vehicles = config["num_vehicles"]  # 车辆数量（5-30）
        self.edge_resource = config["edge_resource"]  # 边缘计算资源（15GHz）
        self.vehicle_resource = config["vehicle_resource"]  # 车端计算资源（1.5GHz）
        self.bandwidth = config["bandwidth"]  # 通信带宽（5-25Mbps）
        self.delay_tolerate = config["delay_tolerate"]  # 容忍延迟（AlexNet25ms/VGG16250ms）
        self.ac_min = config["ac_min"]  # 最小推理精度（0.8）
        self.per_layer_compute = 0.1  # 每层计算量（GOPS，论文3.2节）

        # MDP五要素初始化（论文4.2节）
        self.state_dim = 4  # 状态维度：s(t)=(ac(t), queue_e(t), c_e^rm(t), λ(t))
        self.action_dim = len(model.backbone) + 1  # 动作维度：划分点数量（0到l）
        self.gamma = config["gamma"]  # 折扣因子（0.9，1-147表格）

        # 环境状态初始化
        self.reset()

    def reset(self):
        """重置环境状态（论文4.2节状态空间定义）"""
        # 1. 当前推理精度（ac(t)∈[ac_min,1]）
        self.current_ac = np.random.uniform(self.ac_min, 1.0)
        # 2. 边缘队列长度（queue_e(t)，初始0）
        self.edge_queue = 0
        # 3. 边缘剩余资源（c_e^rm(t)，初始为总资源）
        self.edge_remaining_resource = self.edge_resource
        # 4. 任务到达率（λ(t)∈[15%,25%]，1-147表格）
        self.task_arrival_rate = np.random.uniform(0.15, 0.25)

        # 数据迭代器重置
        self.data_iter = iter(self.dataloader)
        return self._get_state()

    def _get_state(self):
        """获取归一化状态（论文4.2节状态空间）"""
        state = np.array([
            self.current_ac,  # 精度（已归一化）
            min(self.edge_queue / 50, 1.0),  # 队列长度（最大50，归一化到[0,1]）
            self.edge_remaining_resource / self.edge_resource,  # 剩余资源占比
            self.task_arrival_rate / 0.25  # 任务到达率（最大25%，归一化到[0,1]）
        ], dtype=np.float32)
        return torch.tensor(state, dtype=torch.float32)

    def _calculate_reward(self, avg_delay, task_completion_rate, current_ac):
        """
        奖励函数（论文4.2节：R(t) = -delay_avg + α·completion_rate + β·ac）
        :param avg_delay: 平均延迟（ms）
        :param task_completion_rate: 任务完成率
        :param current_ac: 当前精度
        :return: 奖励值
        """
        alpha = 0.5  # 完成率权重
        beta = 0.3   # 精度权重
        # 延迟惩罚（归一化到[-1,0]）
        delay_penalty = -avg_delay / self.delay_tolerate
        # 完成率奖励（[0,0.5]）
        completion_reward = alpha * task_completion_rate
        # 精度奖励（[0,0.3]）
        ac_reward = beta * current_ac
        return delay_penalty + completion_reward + ac_reward

    def step(self, action):
        """
        环境一步交互（论文4.2节MDP转移）
        :param action: 动作（[partition_point, exit_idx]）
        :return: next_state, reward, done, info（延迟、完成率等）
        """
        partition_point, exit_idx = action
        done = False

        # 1. 模型划分（车端+边缘端，论文3.2节）
        vehicle_model, edge_model = self.model.partition_model(partition_point)

        # 2. 加载当前批次数据（BDD100K）
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        imgs = batch["image"].to(self.config["device"])
        labels = batch["label"].to(self.config["device"])

        # 3. 车端推理（含早退出判断，论文3.4节）
        with torch.no_grad():
            vehicle_feat = vehicle_model(imgs)
            if exit_idx != -1 and exit_idx < len(self.model.exit_layers):
                # 早退出：车端直接输出结果
                exit_prob = self.model.exit_layers[exit_idx](vehicle_feat)
                self.current_ac = torch.mean(torch.max(exit_prob, dim=1)[0]).item()
                vehicle_delay = self._calculate_vehicle_delay(partition_point)
                edge_delay = 0.0
                transmission_delay = 0.0
            else:
                # 无早退出：车端→边缘端传输+边缘推理
                transmission_delay = self._calculate_transmission_delay(vehicle_feat)
                edge_feat = edge_model(vehicle_feat)
                main_prob = torch.nn.functional.softmax(edge_feat, dim=1)
                self.current_ac = torch.mean(torch.max(main_prob, dim=1)[0]).item()
                vehicle_delay = self._calculate_vehicle_delay(partition_point)
                edge_delay = self._calculate_edge_delay(partition_point)

        # 4. 计算核心指标（论文5.4节指标）
        total_delay = vehicle_delay + transmission_delay + edge_delay
        avg_delay = calculate_avg_delay(total_delay, self.num_vehicles)
        task_completion_rate = calculate_task_completion_rate(avg_delay, self.delay_tolerate)

        # 5. 更新环境状态（论文4.2节状态转移）
        self.edge_queue = max(0, self.edge_queue + int(self.num_vehicles * self.task_arrival_rate) - 1)
        self.edge_remaining_resource = max(0, self.edge_remaining_resource -
                                           (partition_point / len(self.model.backbone)) * self.edge_resource)
        self.task_arrival_rate = np.clip(self.task_arrival_rate + np.random.normal(0, 0.01), 0.15, 0.25)

        # 6. 计算奖励
        reward = self._calculate_reward(avg_delay, task_completion_rate, self.current_ac)

        # 7. 终止条件（精度低于阈值或资源耗尽，论文约束）
        if self.current_ac < self.ac_min or self.edge_remaining_resource <= 0:
            done = True

        return self._get_state(), reward, done, {
            "avg_delay": avg_delay,
            "task_completion_rate": task_completion_rate,
            "current_ac": self.current_ac
        }

    def _calculate_vehicle_delay(self, partition_point):
        """
        车端延迟计算（论文3.5节延迟模型：delay = 计算量 / 车端资源）
        :param partition_point: 划分点（车端执行层数）
        :return: 车端延迟（ms）
        """
        compute_load = partition_point * self.per_layer_compute  # 总计算量（GOPS）
        delay = (compute_load / self.vehicle_resource) * 1000  # 转换为ms
        return delay

    def _calculate_edge_delay(self, partition_point):
        """
        边缘延迟计算（论文3.5节M/D/1队列模型：处理延迟+队列延迟）
        :param partition_point: 划分点（边缘端执行层数）
        :return: 边缘延迟（ms）
        """
        remaining_layers = len(self.model.backbone) - partition_point
        compute_load = remaining_layers * self.per_layer_compute  # 边缘计算量（GOPS）
        # 服务率（论文公式8：μ(t)=C_e/∑c_j）
        service_rate = self.edge_resource / compute_load  # 任务/ms
        # 到达率（任务/ms）
        arrival_rate = self.task_arrival_rate * self.num_vehicles

        if service_rate <= arrival_rate:
            return float("inf")  # 边缘过载，延迟无穷大

        # M/D/1延迟公式（论文公式8）
        process_delay = 1 / service_rate  # 处理延迟
        queue_delay = arrival_rate / (2 * service_rate * (service_rate - arrival_rate))  # 队列延迟
        return process_delay + queue_delay

    def _calculate_transmission_delay(self, feat):
        """
        传输延迟计算（论文3.6节通信模型：delay = 数据量 / 带宽）
        :param feat: 车端输出中间特征
        :return: 传输延迟（ms）
        """
        # 特征数据量（MB，float32=4字节）
        data_size = feat.nelement() * 4 / (1024 * 1024)
        # 传输延迟（ms，带宽单位：Mbps）
        delay = (data_size * 8) / self.bandwidth * 1000
        return delay