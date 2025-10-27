from typing import Dict, Any, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from model.base_model import BaseMultiExitModel
from adp_d3qn.model_partition import get_edge_model_layers


class EdgeInference:
    """
    边缘端推理器（Edge-side）
    --------------------------------------------------------------
    对应论文 Section IV-D：
      - 划分点之后的剩余层在边缘端执行
      - M/D/1 队列延迟模型 (Eq.8): delay = 1/μ + λ/[2μ(μ-λ)]
      - 服务率 μ(t) = C_e / (Σ c_j)  (剩余层计算量)
      - 到达率 λ(t) 由系统演化/入队过程驱动
    """

    def __init__(self, model: BaseMultiExitModel, config: Dict[str, Any]):
        self.model = model
        self.device = torch.device(config.get("edge_device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

        # === 参数全来自 config 或模型 ===
        self.edge_resource = float(config.get("edge_resource", 15.0))               # C_e (GHz)
        self.queue_capacity = int(config.get("edge_queue_capacity", 50))            # 队列容量（监控指标）
        self.arrival_low, self.arrival_high = tuple(config.get("arrival_rate_range", (0.15, 0.25)))
        self.arrival_sigma = float(config.get("arrival_rate_sigma", 0.01))          # 到达率噪声
        self.ac_min = float(config.get("ac_min", 0.8))
        self.per_layer_compute = float(config.get("per_layer_compute", 0.1))        # GOPS（若无 layer-wise 统计时的兜底）
        self.energy_factor = float(config.get("edge_energy_per_gop", 0.04))         # J/GOP
        self.initial_partition_point = int(config.get("initial_partition_point", 0))

        # 队列与到达率
        self.task_queue: deque = deque(maxlen=self.queue_capacity)
        self.task_arrival_rate = float(config.get("initial_task_arrival_rate", (self.arrival_low + self.arrival_high) / 2))

        # 当前划分点与服务率
        self.current_partition_point = self.initial_partition_point
        self.edge_model = self._build_edge_model(self.current_partition_point)       # 仅剩余层与最终头
        self.edge_model.to(self.device).eval()
        self.service_rate = self._recompute_service_rate(self.current_partition_point)  # μ

    # ------------------------------------------------------------------
    # 构建边缘端子模型：划分点之后的层 + 最终分类头
    # ------------------------------------------------------------------
    def _build_edge_model(self, partition_point: int) -> nn.Module:
        """将主干剩余层拼成一个顺序模块；最终的主出口由 backbone/head 内部实现。"""
        if partition_point < 0 or partition_point > len(self.model.backbone):
            raise ValueError(f"Partition point must be in [0, {len(self.model.backbone)}].")

        # 仅收集划分点之后的 backbone 层
        edge_layers = get_edge_model_layers(self.model.backbone, partition_point)
        # 如果 base_model 的 forward_from 支持从中间层继续，则这里也可只保存 partition_point
        return nn.Sequential(*edge_layers)

    def update_partition_point(self, new_partition_point: int) -> None:
        """根据 ADP-D3QN 动作更新划分点，重建子模型并更新服务率 μ"""
        self.current_partition_point = int(new_partition_point)
        self.edge_model = self._build_edge_model(self.current_partition_point).to(self.device).eval()
        self.service_rate = self._recompute_service_rate(self.current_partition_point)

    # ------------------------------------------------------------------
    # 资源/服务率计算 —— μ = C_e / (Σ c_j)   (Eq.8 的组成项)
    # ------------------------------------------------------------------
    def _recompute_service_rate(self, partition_point: int) -> float:
        """根据剩余层计算量重估 μ（任务/秒），避免写死常数"""
        remaining_layers = max(0, len(self.model.backbone) - partition_point)

        # 若模型提供逐层计算量统计则优先使用（与 model_partition.get_model_layer_compute_cost 一致）
        if hasattr(self.model, "layer_compute") and isinstance(self.model.layer_compute, dict):
            total_compute_gops = float(sum(self.model.layer_compute.get(i, 0.0) for i in range(partition_point, len(self.model.backbone))))
        else:
            # 兜底：每层 per_layer_compute（来自 config），严格不写死在代码里
            total_compute_gops = float(remaining_layers) * self.per_layer_compute

        # μ ≈ C_e / 计算量（简化“每任务计算量”，单位对齐到 “任务/秒”的近似）
        # 注意：这里是宏观仿真近似，供审稿复现实验，不代表真实吞吐测量。
        mu = self.edge_resource / max(total_compute_gops, 1e-9)
        return float(mu)

    # ------------------------------------------------------------------
    # 入队：车辆上传的中间特征
    # ------------------------------------------------------------------
    def add_task_to_queue(self, task: Dict[str, Any]) -> bool:
        """
        task 需要包含：
          - "feat": np.ndarray 或 torch.Tensor (B,C,H,W)
          - "meta": 可选，车辆ID、时间戳等
        """
        if len(self.task_queue) >= self.queue_capacity:
            return False
        self.task_queue.append(task)

        # 更新 λ：加入轻微噪声（来自 config 的 sigma），并裁剪在范围内
        self.task_arrival_rate = float(np.clip(
            self.task_arrival_rate + np.random.normal(0.0, self.arrival_sigma),
            self.arrival_low, self.arrival_high
        ))
        return True

    # ------------------------------------------------------------------
    # 边缘推理：出队 → 剩余层前向 → 可选早退 → 主出口
    # ------------------------------------------------------------------
    def infer(self, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        从队列取一个任务并完成边缘推理
        :param labels: 可选标签（B,），仅用于计算精度展示
        """
        if not self.task_queue:
            # 无任务；可返回队列状态
            return {"status": "empty", "delay": 0.0, "accuracy": 0.0, "queue_len": len(self.task_queue)}

        task = self.task_queue.popleft()
        feat = task["feat"]
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        feat = feat.to(self.device, dtype=torch.float32)

        # 计时（仅用于记录程序执行时间；真正的 M/D/1 延迟由公式给出）
        wall_t0 = time.time()

        with torch.no_grad():
            # 1) 剩余层前向（backbone 的后半段）
            edge_feat = self.edge_model(feat)

            # 2) 边缘侧可选早退：若模型设计允许在边缘侧仍有出口，则对“划分点后的出口”逐一判定
            exit_idx_used = -1
            confidence = 0.0
            if hasattr(self.model, "exit_layers") and len(self.model.exit_layers) > 0:
                # 将划分点映射为“从哪个出口开始可用”
                first_exit = self._map_partition_to_first_exit()
                for e_idx in range(first_exit, len(self.model.exit_layers)):
                    logits_e = self.model.exit_layers[e_idx](edge_feat)
                    probs_e = F.softmax(logits_e, dim=1)
                    conf_e, pred_e = torch.max(probs_e, dim=1)
                    conf_mean = float(conf_e.mean().item())
                    if conf_mean >= self.ac_min:
                        # 触发边缘侧早退
                        exit_idx_used = e_idx
                        confidence = conf_mean
                        final_probs = probs_e
                        final_pred = pred_e
                        break

            # 3) 若未在任何出口早退，则使用“主出口”（模型的最终头）
            if exit_idx_used == -1:
                # 假设 base_model 提供 forward_from() 执行剩余 head
                if hasattr(self.model, "forward_from"):
                    logits_main = self.model.forward_from(edge_feat, start_layer=len(self.model.backbone))  # 或者 model.head(edge_feat)
                else:
                    # 某些实现中 edge_model 已经含有最终 head；这里做兼容
                    logits_main = edge_feat
                final_probs = F.softmax(logits_main, dim=1)
                final_pred = final_probs.argmax(dim=1)
                confidence = float(final_probs.max(dim=1).values.mean().item())

        # 4) 计算 M/D/1 延迟 (Eq.8)
        process_delay_ms = self._process_delay_ms()      # 1/μ
        queue_delay_ms = self._queue_delay_ms()          # λ/[2μ(μ-λ)]
        model_time_ms = (time.time() - wall_t0) * 1000.0 # 实测代码时间（仅展示）
        total_delay_ms = process_delay_ms + queue_delay_ms + model_time_ms

        # 5) 能耗估算（与车辆端一致：计算量 × 因子）
        energy_j = self._edge_energy_j()

        # 6) 可选精度（若提供标签）
        accuracy = 0.0
        if labels is not None:
            labels = labels.to(self.device)
            accuracy = float((final_pred == labels).float().mean().item())

        return {
            "status": "completed",
            "exit_idx": exit_idx_used,                 # -1 表示主出口
            "confidence": confidence,
            "delay": float(total_delay_ms),
            "delay_breakdown": {
                "process_ms": float(process_delay_ms),
                "queue_ms": float(queue_delay_ms),
                "overhead_ms": float(model_time_ms)
            },
            "energy": float(energy_j),
            "accuracy": float(accuracy),
            "queue_len": len(self.task_queue),
            "partition_point": int(self.current_partition_point)
        }

    # ------------------------- 内部计算 -------------------------

    def _map_partition_to_first_exit(self) -> int:
        """把划分点映射为“从哪个出口开始可用”的索引"""
        n_layers = len(self.model.backbone)
        n_exits = len(getattr(self.model, "exit_layers", []))
        if n_exits == 0:
            return 0
        ratio = self.current_partition_point / max(1, n_layers)
        first_exit = int(np.floor(ratio * n_exits))
        return min(max(first_exit, 0), n_exits - 1)

    def _process_delay_ms(self) -> float:
        """处理延迟：1/μ（秒）→ ms"""
        if self.service_rate <= 0:
            return float("inf")
        return float((1.0 / self.service_rate) * 1000.0)

    def _queue_delay_ms(self) -> float:
        """队列延迟：λ / [2μ(μ-λ)]（秒）→ ms"""
        mu = self.service_rate
        lam = self.task_arrival_rate
        if mu <= lam:
            return float("inf")
        return float((lam / (2.0 * mu * (mu - lam))) * 1000.0)

    def _edge_energy_j(self) -> float:
        """能耗估计：Σ c_j × 因子（J/GOP）"""
        remaining_layers = max(0, len(self.model.backbone) - self.current_partition_point)
        if hasattr(self.model, "layer_compute") and isinstance(self.model.layer_compute, dict):
            total_compute_gops = float(sum(self.model.layer_compute.get(i, 0.0) for i in range(self.current_partition_point, len(self.model.backbone))))
        else:
            total_compute_gops = float(remaining_layers) * self.per_layer_compute
        return float(total_compute_gops * self.energy_factor)


# ------------------------------ 简单测试 ------------------------------
if __name__ == "__main__":
    from model.vgg16 import MultiExitVGG16

    cfg = {
        "edge_device": "cuda" if torch.cuda.is_available() else "cpu",
        "edge_resource": 15.0,
        "edge_queue_capacity": 50,
        "arrival_rate_range": (0.15, 0.25),
        "arrival_rate_sigma": 0.01,
        "initial_task_arrival_rate": 0.2,
        "initial_partition_point": 5,
        "ac_min": 0.8,
        "per_layer_compute": 0.1,
        "edge_energy_per_gop": 0.04,
    }

    model = MultiExitVGG16(num_classes=10).to(cfg["edge_device"])
    inferer = EdgeInference(model, cfg)

    # 模拟入队一个任务（Block2 后特征尺寸示例）
    fake_feat = np.random.randn(1, 128, 56, 56).astype(np.float32)
    inferer.add_task_to_queue({"feat": fake_feat, "meta": {"vehicle_id": 1}})

    # 推理
    out = inferer.infer()
    print("[Edge] result:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "delay_breakdown"})
    print("[Edge] delay breakdown (ms):", {k: round(v, 3) for k, v in out["delay_breakdown"].items()})
