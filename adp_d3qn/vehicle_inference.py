import torch
import time
import numpy as np
from model.base_model import BaseMultiExitModel
from adp_d3qn.metrics import calculate_inference_accuracy


class VehicleInference:
    def __init__(self, model: BaseMultiExitModel, config):
        """
        车端推理器（本地计算+早退出判断）
        :param model: 多出口DNN模型
        :param config: 配置字典
        """
        self.model = model
        self.device = config["device"] if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # 推理模式
        self.partition_point = config["initial_partition_point"]  # 初始划分点
        self.ac_min = config["ac_min"]  # 精度阈值
        self.vehicle_resource = config["vehicle_resource"]  # 车端计算资源（GHz）

        # 加载车端子模型（初始划分点）
        self.vehicle_model, _ = self.model.partition_model(self.partition_point)
        self.vehicle_model.to(self.device)
        self.vehicle_model.eval()

    def update_partition_point(self, new_partition_point):
        """更新模型划分点并重新加载车端子模型"""
        if new_partition_point < 0 or new_partition_point > len(self.model.backbone.layers):
            raise ValueError(f"Partition point must be in [0, {len(self.model.backbone.layers)}]")
        self.partition_point = new_partition_point
        self.vehicle_model, _ = self.model.partition_model(self.partition_point)
        self.vehicle_model.to(self.device)
        self.vehicle_model.eval()

    def infer(self, img):
        """
        车端推理（含早退出判断）
        :param img: 输入图像（np.ndarray, (C,H,W)）
        :return: 推理结果（含是否早退出、延迟、精度）
        """
        # 图像预处理
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 推理计时
        start_time = time.time()

        # 车端前向计算
        with torch.no_grad():
            vehicle_feat = self.vehicle_model(img)

            # 早退出判断（仅当划分点>0且有早退出层）
            if self.partition_point > 0 and self.model.exit_layers:
                # 获取当前划分点对应的早退出层索引
                exit_idx = min(self.partition_point // len(self.model.backbone.layers) * len(self.model.exit_layers),
                               len(self.model.exit_layers) - 1)
                exit_prob = self.model.exit_layers[exit_idx](vehicle_feat)
                exit_acc = calculate_inference_accuracy(exit_prob, self.labels)  # 需传入标签计算精度

                # 早退出决策
                if exit_acc >= self.ac_min:
                    # 早退出：返回车端结果
                    end_time = time.time()
                    delay = (end_time - start_time) * 1000  # 转换为ms
                    return {
                        "early_exit": True,
                        "exit_idx": exit_idx,
                        "delay": delay,
                        "accuracy": exit_acc,
                        "result": torch.argmax(exit_prob, dim=1).item()
                    }
                else:
                    # 不早退出：返回中间特征（需传输到边缘端）
                    end_time = time.time()
                    delay = (end_time - start_time) * 1000
                    return {
                        "early_exit": False,
                        "exit_idx": -1,
                        "delay": delay,
                        "accuracy": 0.0,  # 未完成推理
                        "intermediate_feat": vehicle_feat.cpu().numpy()
                    }
            else:
                # 无早退出层：返回中间特征
                end_time = time.time()
                delay = (end_time - start_time) * 1000
                return {
                    "early_exit": False,
                    "exit_idx": -1,
                    "delay": delay,
                    "accuracy": 0.0,
                    "intermediate_feat": vehicle_feat.cpu().numpy()
                }


if __name__ == "__main__":
    # 测试代码
    from config import Config
    from model.vgg16 import MultiExitVGG16
    from dataset.bdd100k_processor import get_bdd100k_dataloader

    config = Config()
    model = MultiExitVGG16(num_classes=config.num_classes, ac_min=config.ac_min)
    dataloader = get_bdd100k_dataloader(
        data_root=config.data_root,
        split="val",
        batch_size=1,
        img_size=config.img_size,
        augment=False
    )

    # 车端推理器初始化
    vehicle_infer = VehicleInference(model=model, config=config.__dict__)
    vehicle_infer.update_partition_point(partition_point=10)  # Block 2后划分

    # 单样本推理
    for batch in dataloader:
        img = batch["image"][0].numpy()  # (C,H,W)
        label = batch["label"][0].item()
        vehicle_infer.labels = torch.tensor([label], device=vehicle_infer.device)  # 传入标签

        result = vehicle_infer.infer(img)
        print("Vehicle Inference Result:", result)
        break