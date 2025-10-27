import torch
import torch.nn as nn


class BaseMultiExitModel(nn.Module):
    """
    多出口DNN模型基类（论文§3.2–3.4）
    - 支持统一forward与early-exit机制
    - 兼容模型划分与边缘推理
    """

    def __init__(self, num_classes=7, ac_min=0.8):
        super().__init__()
        self.num_classes = num_classes
        self.ac_min = ac_min
        self.exit_layers = nn.ModuleList()
        self.backbone = None
        self.layer_compute_cost = {}

    # ------------------------------- 注册接口 ------------------------------- #
    def register_exit_layer(self, exit_layer: nn.Module):
        """注册早退出层（论文§3.4）"""
        self.exit_layers.append(exit_layer)

    # ------------------------------- 前向传播 ------------------------------- #
    def forward(self, x, return_exit_probs=False, cache_features=False):
        """
        统一前向传播逻辑
        :param x: 输入图像 (B,C,H,W)
        :param return_exit_probs: 是否返回所有出口概率
        :param cache_features: 是否缓存中间特征（便于模型划分）
        :return: main_prob, exit_probs(可选), features(可选)
        """
        exit_probs, features = [], []
        feat = x

        for layer_idx, layer in enumerate(self.backbone):
            feat = layer(feat)
            if cache_features:
                features.append(feat)
            # 判断是否到达早退出挂钩点
            if hasattr(self, "exit_hook_points") and layer_idx in self.exit_hook_points:
                hook_idx = self.exit_hook_points.index(layer_idx)
                if hook_idx < len(self.exit_layers):
                    out = self.exit_layers[hook_idx](feat)
                    exit_probs.append(nn.functional.softmax(out, dim=1))

        main_prob = nn.functional.softmax(feat, dim=1)

        if return_exit_probs and cache_features:
            return main_prob, exit_probs, features
        elif return_exit_probs:
            return main_prob, exit_probs
        elif cache_features:
            return main_prob, features
        else:
            return main_prob

    # ------------------------------- 早退出决策 ------------------------------- #
    def get_early_exit_decision(self, exit_probs):
        """
        根据论文Eq.(6)与Eq.(15)，判断是否提前退出
        :param exit_probs: 各出口预测概率
        :return: (exit_idx, confidence)
        """
        if not exit_probs:
            return -1, 0.0
        for idx, prob in enumerate(exit_probs):
            conf, _ = torch.max(prob, dim=1)
            avg_conf = conf.mean().item()
            if avg_conf >= self.ac_min:
                return idx, avg_conf
        return -1, 0.0

    # ------------------------------- 模型划分 ------------------------------- #
    def partition_model(self, partition_point):
        """
        按划分点拆分为车端/边缘模型（论文§3.2 Eq.(15d)）
        :return: (vehicle_model, edge_model)
        """
        if partition_point < 0 or partition_point > len(self.backbone):
            raise ValueError(f"Partition point must be in [0, {len(self.backbone)}]")

        vehicle_layers = self.backbone[:partition_point]
        edge_layers = self.backbone[partition_point:]

        vehicle_model = nn.Sequential(*vehicle_layers)
        edge_model = nn.Sequential(*edge_layers, *self.exit_layers)
        return vehicle_model, edge_model

    # ------------------------------- 计算量估计 ------------------------------- #
    def calculate_layer_compute_cost(self):
        """
        计算每层计算量 (GOPS)，用于资源分配与时延建模（论文§3.2）
        """
        self.layer_compute_cost = {}
        for idx, layer in enumerate(self.backbone):
            if isinstance(layer, nn.Conv2d):
                Cin, Cout, kh, kw = layer.weight.shape
                H = 224 // (2 ** (idx // 5))
                W = H
                flops = 2 * Cin * Cout * kh * kw * H * W / 1e9
            elif isinstance(layer, nn.Linear):
                flops = 2 * layer.in_features * layer.out_features / 1e9
            else:
                flops = 0.01
            self.layer_compute_cost[idx] = flops
            setattr(layer, "compute_cost", flops)
        print(f"[Layer Cost] Computed {len(self.layer_compute_cost)} layers (unit: GOPS)")
        return self.layer_compute_cost

    # ------------------------------- 信息打印 ------------------------------- #
    def summary(self):
        """打印模型结构摘要"""
        print("=" * 80)
        print(f"[Model Summary] {self.__class__.__name__}")
        print(f" - Classes: {self.num_classes}")
        print(f" - Accuracy threshold: {self.ac_min}")
        print(f" - Backbone layers: {len(self.backbone)}")
        print(f" - Exit layers: {len(self.exit_layers)} at {getattr(self, 'exit_hook_points', [])}")
        print("=" * 80)
