import torch
import torch.nn.functional as F
import time
import numpy as np
from model.base_model import BaseMultiExitModel


class VehicleInference:
    """
    车端推理器（Vehicle-side Inference）
    --------------------------------------------------------------
    对应论文 Section IV-C：
      - Eq.(1): softmax-based early-exit confidence
      - Eq.(2): exit decision threshold
      - Eq.(3): exit probability propagation
      - Eq.(9): vehicle-side latency model
    """

    def __init__(self, model: BaseMultiExitModel, config: dict):
        self.model = model
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

        # ==== 动态参数 ====
        self.partition_point = config.get("initial_partition_point", 0)
        self.ac_min = config.get("ac_min", 0.8)
        self.vehicle_resource = config.get("vehicle_resource", 1.5)  # GHz
        self.per_layer_compute = config.get("per_layer_compute", 0.1)  # GOPS
        self.energy_factor = config.get("vehicle_energy_per_gop", 0.05)  # J/GOP

        # 初始化子模型（划分点可变）
        self._load_vehicle_model()

    # --------------------------------------------------------------
    # Section IV-C-1: 更新划分点
    # --------------------------------------------------------------
    def update_partition_point(self, new_partition_point: int):
        """动态更新模型划分点"""
        if new_partition_point < 0 or new_partition_point > len(self.model.backbone):
            raise ValueError(f"Invalid partition point: {new_partition_point}")
        self.partition_point = new_partition_point
        self._load_vehicle_model()

    def _load_vehicle_model(self):
        """加载车端子模型"""
        self.vehicle_model, _ = self.model.partition_model(self.partition_point)
        self.vehicle_model.to(self.device).eval()

    # --------------------------------------------------------------
    # Section IV-C-2: 车端早退出判定逻辑 (Eq.1–4)
    # --------------------------------------------------------------
    def infer(self, img: torch.Tensor):
        """
        执行车端推理，包含早退出判断
        :param img: 输入图像 Tensor[C,H,W]
        :return: dict {early_exit, exit_idx, delay, energy, accuracy/confidence, result/intermediate_feat}
        """
        img = img.unsqueeze(0).to(self.device, dtype=torch.float32)
        start_t = time.time()

        with torch.no_grad():
            # Step 1. 前向传播到划分点
            feat = self.vehicle_model(img)

            # Step 2. 检查是否存在可访问的出口层
            if self.partition_point > 0 and hasattr(self.model, "exit_layers") and len(self.model.exit_layers) > 0:
                # 映射划分点 → 出口层索引
                exit_idx = self._map_partition_to_exit()
                exit_layer = self.model.exit_layers[exit_idx]

                # Step 3. Eq.(1): softmax 输出置信度
                logits = exit_layer(feat)
                probs = F.softmax(logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)
                conf_mean = float(confidence.mean().item())

                # Step 4. Eq.(2): 判定是否满足早退出条件
                if conf_mean >= self.ac_min:
                    # ---- Early Exit Triggered ----
                    delay_ms, energy_j = self._compute_vehicle_cost()
                    return {
                        "early_exit": True,
                        "exit_idx": exit_idx,
                        "delay": delay_ms,
                        "energy": energy_j,
                        "confidence": conf_mean,
                        "result": int(pred.item())
                    }

                # ---- 未早退：传输中间特征 ----
                delay_ms, energy_j = self._compute_vehicle_cost()
                return {
                    "early_exit": False,
                    "exit_idx": -1,
                    "delay": delay_ms,
                    "energy": energy_j,
                    "confidence": conf_mean,
                    "intermediate_feat": feat.detach().cpu().numpy()
                }

            else:
                # 无早退出层（直接传输）
                delay_ms, energy_j = self._compute_vehicle_cost()
                return {
                    "early_exit": False,
                    "exit_idx": -1,
                    "delay": delay_ms,
                    "energy": energy_j,
                    "confidence": 0.0,
                    "intermediate_feat": feat.detach().cpu().numpy()
                }

    # --------------------------------------------------------------
    # Section IV-C-3: 辅助函数
    # --------------------------------------------------------------
    def _map_partition_to_exit(self) -> int:
        """
        动态映射划分点 → 出口层索引
        Eq.(3): exit_idx ≈ floor(par / l * n_exit)
        """
        n_layers = len(self.model.backbone)
        n_exits = len(self.model.exit_layers)
        if n_exits == 0:
            return -1
        ratio = self.partition_point / max(1, n_layers)
        exit_idx = int(np.floor(ratio * n_exits))
        return min(exit_idx, n_exits - 1)

    def _compute_vehicle_cost(self):
        """
        计算车端延迟与能耗
        Eq.(9): delay_icv = (Σ_{j=1}^{par} c_j) / C_i
        其中 c_j≈per_layer_compute, C_i=vehicle_resource(GHz)
        """
        compute_load = self.partition_point * self.per_layer_compute  # GOPS
        delay_s = compute_load / max(self.vehicle_resource, 1e-9)
        delay_ms = delay_s * 1000.0
        energy_j = compute_load * self.energy_factor
        return delay_ms, energy_j


# --------------------------------------------------------------
# 测试 (简版)
# --------------------------------------------------------------
if __name__ == "__main__":
    from model.alexnet import MultiExitAlexNet
    from dataset.bdd100k_processor import get_bdd100k_dataloader
    from config import Config

    config = Config()
    config_dict = config.__dict__

    model = MultiExitAlexNet(num_classes=config.num_classes, ac_min=config.ac_min)
    dataloader = get_bdd100k_dataloader(
        data_root=config.data_root,
        split="val",
        batch_size=1,
        img_size=config.img_size,
        augment=False
    )

    veh_infer = VehicleInference(model=model, config=config_dict)
    veh_infer.update_partition_point(new_partition_point=10)

    for batch in dataloader:
        img = batch["image"][0]
        result = veh_infer.infer(img)
        print("Vehicle-side inference result:", result)
        break
