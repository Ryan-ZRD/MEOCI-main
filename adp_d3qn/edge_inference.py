import torch
import time
import numpy as np
from collections import deque
from model.base_model import BaseMultiExitModel
from adp_d3qn.metrics import calculate_inference_accuracy
from adp_d3qn.model_partition import get_edge_model_layers


class EdgeInference:
    def __init__(self, model: BaseMultiExitModel, config):
        """
        边缘端推理器（论文M/D/1队列模型+剩余层计算）
        :param model: 多出口DNN模型（论文中AlexNet/VGG16）
        :param config: 配置字典（含边缘资源、队列容量等论文参数）
        """
        self.model = model
        self.device = config["edge_device"]  # 边缘端设备（论文用GPU加速）
        self.model.to(self.device)
        self.model.eval()  # 推理模式

        # 论文核心参数
        self.edge_resource = config["edge_resource"]  # 边缘计算资源（论文中15GHz）
        self.queue_capacity = config["edge_queue_capacity"]  # 队列容量（论文设为50）
        self.service_rate = 0.0  # 服务率（论文公式8：μ(t)=C_e/∑c_j）
        self.task_arrival_rate = config["initial_task_arrival_rate"]  # 初始任务到达率（15%-25%）

        # M/D/1队列初始化（论文3.5节延迟模型）
        self.task_queue = deque(maxlen=self.queue_capacity)
        self.current_partition_point = config["initial_partition_point"]  # 初始划分点（论文默认0<par<l）

        # 加载边缘端子模型（基于当前划分点）
        self.edge_model = self._get_edge_model(self.current_partition_point)
        self.edge_model.to(self.device)
        self.edge_model.eval()

    def _get_edge_model(self, partition_point):
        """
        基于论文模型划分逻辑，获取边缘端子模型（剩余层+出口层）
        :param partition_point: 划分点（par(t)∈{0,1,...,l}）
        :return: 边缘端子模型
        """
        if partition_point < 0 or partition_point > len(self.model.backbone.layers):
            raise ValueError(f"Partition point must be in [0, {len(self.model.backbone.layers)}] (Paper Eq.15d)")

        # 论文逻辑：边缘端执行划分点后的所有层 + 所有早退出层
        edge_layers = get_edge_model_layers(self.model.backbone.layers, partition_point)
        edge_model = nn.Sequential(*edge_layers, *self.model.exit_layers)
        return edge_model

    def update_partition_point(self, new_partition_point):
        """
        根据ADP-D3QN决策更新划分点（论文4.2节动作空间）
        :param new_partition_point: 新划分点（由智能体输出）
        """
        self.current_partition_point = new_partition_point
        self.edge_model = self._get_edge_model(new_partition_point)
        self.edge_model.to(self.device)
        self.edge_model.eval()

        # 更新服务率（论文公式8：μ(t)=C_e/∑c_j，c_j为边缘层计算量）
        edge_layer_compute = sum([layer.compute_cost for layer in self.edge_model])  # 每层计算量（论文预定义）
        self.service_rate = self.edge_resource / edge_layer_compute  # 服务率（任务/ms）

    def add_task_to_queue(self, task_data):
        """
        将车端传输的中间特征加入队列（论文3.5节任务堆叠处理）
        :param task_data: 任务数据（中间特征+车辆ID+时间戳）
        :return: 是否成功入队
        """
        if len(self.task_queue) < self.queue_capacity:
            self.task_queue.append(task_data)
            # 更新任务到达率（论文中随机波动±1%）
            self.task_arrival_rate = np.clip(self.task_arrival_rate + np.random.normal(0, 0.01), 0.15, 0.25)
            return True
        return False  # 队列满，任务丢弃（论文中计入任务失败率）

    def _calculate_queue_delay(self):
        """
        计算队列延迟（论文M/D/1模型公式8：λ/(2μ(μ-λ))）
        :return: 队列延迟（ms）
        """
        if self.service_rate <= self.task_arrival_rate:
            return float("inf")  # 边缘过载，延迟无穷大
        queue_delay = self.task_arrival_rate / (2 * self.service_rate * (self.service_rate - self.task_arrival_rate))
        return queue_delay

    def _calculate_process_delay(self):
        """
        计算边缘端处理延迟（论文公式8：1/μ(t)）
        :return: 处理延迟（ms）
        """
        return 1.0 / self.service_rate if self.service_rate > 0 else float("inf")

    def infer(self, intermediate_feat, labels=None):
        """
        边缘端推理（论文流程：出队→计算→早退出判断→返回结果）
        :param intermediate_feat: 车端传输的中间特征（论文中d_l大小）
        :param labels: 标签（用于计算精度，论文中BDD100K标签）
        :return: 推理结果（含延迟、精度、是否主出口）
        """
        # 1. 任务出队（若队列为空，等待新任务）
        if not self.task_queue:
            time.sleep(0.001)  # 模拟等待
            return {"status": "waiting", "delay": 0.0, "accuracy": 0.0}

        task = self.task_queue.popleft()
        start_time = time.time()

        # 2. 边缘端前向计算（剩余层）
        with torch.no_grad():
            intermediate_feat = torch.tensor(intermediate_feat, dtype=torch.float32).to(self.device)
            edge_feat = self.edge_model(intermediate_feat)

            # 3. 早退出判断（论文3.4节：基于出口概率pr_i^early）
            main_prob = torch.nn.functional.softmax(edge_feat, dim=1)
            exit_probs = self.model.exit_layers  # 各早退出层概率（论文公式6）
            exit_idx, exit_acc = self.model.get_early_exit_decision(exit_probs)

            # 4. 若早退出不满足，使用主出口（论文定义主出口为最后一层）
            if exit_idx == -1:
                final_prob = main_prob
                final_acc = calculate_inference_accuracy(final_prob, labels)
                is_main_exit = True
            else:
                final_prob = exit_probs[exit_idx]
                final_acc = exit_acc
                is_main_exit = False

        # 5. 计算总延迟（论文公式8：处理延迟+队列延迟）
        process_delay = self._calculate_process_delay()
        queue_delay = self._calculate_queue_delay()
        total_delay = (time.time() - start_time) * 1000 + process_delay + queue_delay  # 实际时间+模型延迟

        return {
            "status": "completed",
            "is_main_exit": is_main_exit,
            "exit_idx": exit_idx,
            "delay": total_delay,
            "accuracy": final_acc,
            "result": torch.argmax(final_prob, dim=1).cpu().numpy()
        }

    def get_queue_status(self):
        """返回队列状态（论文监控指标：队列长度、任务到达率、服务率）"""
        return {
            "queue_length": len(self.task_queue),
            "task_arrival_rate": self.task_arrival_rate,
            "service_rate": self.service_rate,
            "current_partition_point": self.current_partition_point
        }


if __name__ == "__main__":
    # 论文参数配置（基于1-147表格）
    config = {
        "edge_device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "edge_resource": 15.0,  # 边缘计算资源（GHz）
        "edge_queue_capacity": 50,  # 队列容量
        "initial_task_arrival_rate": 0.2,  # 初始任务到达率20%
        "initial_partition_point": 5,  # 初始划分点（VGG16共31层，取第5层后）
        "num_classes": 10,  # 论文中BDD100K分类任务
        "ac_min": 0.8  # 最小精度阈值（论文Eq.15）
    }

    # 加载论文多出口VGG16模型
    from model.vgg16 import MultiExitVGG16

    model = MultiExitVGG16(num_classes=config["num_classes"], ac_min=config["ac_min"])

    # 初始化边缘推理器
    edge_infer = EdgeInference(model=model, config=config)

    # 模拟车端传输中间特征（随机生成，尺寸匹配VGG16划分点5后的输入）
    intermediate_feat = np.random.randn(1, 128, 56, 56).astype(np.float32)  # VGG16 Block2后特征
    edge_infer.add_task_to_queue({"feat": intermediate_feat, "vehicle_id": 1, "timestamp": time.time()})

    # 边缘端推理
    result = edge_infer.infer(intermediate_feat, labels=np.array([3]))  # 模拟标签3
    print("Edge Inference Result (Paper VGG16):")
    print(
        f"Total Delay: {result['delay']:.2f}ms | Accuracy: {result['accuracy']:.4f} | Main Exit: {result['is_main_exit']}")
    print("Queue Status:", edge_infer.get_queue_status())