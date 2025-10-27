import torch
import torch.nn as nn


class BaseMultiExitModel(nn.Module):
    """
    多出口模型基类（论文3.2-3.4节多出口DNN模型定义）
    统一实现早退出接口、模型划分方法，适配AlexNet/VGG16
    """

    def __init__(self, num_classes=7, ac_min=0.8):
        """
        :param num_classes: 类别数（论文分类任务7类场景）
        :param ac_min: 最小推理精度阈值（论文Eq.15：ac_min=0.8）
        """
        super().__init__()
        self.num_classes = num_classes
        self.ac_min = ac_min  # 早退出精度阈值
        self.exit_layers = nn.ModuleList()  # 早退出层列表（子类注册）
        self.backbone = None  # 主干网络（子类实现，如AlexNet/VGG16）
        self.layer_compute_cost = None  # 每层计算量（论文3.2节模型划分基础）

    def register_exit_layer(self, exit_layer):
        """注册早退出层（论文3.4节多出口设计）"""
        self.exit_layers.append(exit_layer)

    def forward(self, x, return_exit_probs=False):
        """
        前向传播（支持返回各出口概率，论文3.4节推理流程）
        :param x: 输入特征（(B,C,H,W)）
        :param return_exit_probs: 是否返回各早退出层概率
        :return: 主出口概率 + （可选）各早退出层概率列表
        """
        exit_probs = []
        feat = x

        # 主干网络前向，在指定位置输出早退出特征
        for layer_idx, layer in enumerate(self.backbone):
            feat = layer(feat)
            # 检查当前层是否为早退出挂钩点（子类定义）
            if hasattr(self, "exit_hook_points") and layer_idx in self.exit_hook_points:
                hook_idx = self.exit_hook_points.index(layer_idx)
                if hook_idx < len(self.exit_layers):
                    # 早退出层前向，计算概率
                    exit_feat = self.exit_layers[hook_idx](feat)
                    exit_prob = nn.functional.softmax(exit_feat, dim=1)
                    exit_probs.append(exit_prob)

        # 主出口概率（主干网络最后一层输出）
        main_prob = nn.functional.softmax(feat, dim=1)

        if return_exit_probs:
            return main_prob, exit_probs
        return main_prob

    def get_early_exit_decision(self, exit_probs):
        """
        早退出决策（论文3.4节：基于精度阈值ac_min）
        :param exit_probs: 各早退出层概率列表（(B, num_classes)）
        :return: exit_idx（早退出层索引，-1表示主出口）、acc（对应精度）
        """
        for idx, prob in enumerate(exit_probs):
            # 计算当前出口平均精度（最大概率均值）
            max_prob, _ = torch.max(prob, dim=1)
            avg_acc = torch.mean(max_prob).item()
            if avg_acc >= self.ac_min:
                return idx, avg_acc  # 满足精度，早退出
        # 所有早退出层不满足，使用主出口
        main_prob = self.forward(torch.randn(1, 3, 224, 224).to(next(self.parameters()).device))
        main_max_prob, _ = torch.max(main_prob, dim=1)
        main_avg_acc = torch.mean(main_max_prob).item()
        return -1, main_avg_acc

    def partition_model(self, partition_point):
        """
        模型划分（论文3.2节：车端执行前N层，边缘端执行剩余层）
        :param partition_point: 划分点（0=全边缘，l=全车端，0<N<l=分层执行）
        :return: 车端子模型、边缘端子模型
        """
        if partition_point < 0 or partition_point > len(self.backbone):
            raise ValueError(f"Partition point must be in [0, {len(self.backbone)}] (Paper Eq.15d)")

        # 车端子模型：前partition_point层
        vehicle_layers = self.backbone[:partition_point]
        vehicle_model = nn.Sequential(*vehicle_layers)

        # 边缘端子模型：剩余层 + 所有早退出层（论文协同推理逻辑）
        edge_layers = self.backbone[partition_point:]
        edge_model = nn.Sequential(*edge_layers, *self.exit_layers)

        return vehicle_model, edge_model

    def calculate_layer_compute_cost(self):
        """
        计算每层计算量（论文3.2节：用于延迟模型与资源约束）
        :return: 每层计算量字典（key=层索引，value=计算量GOPS）
        """
        self.layer_compute_cost = {}
        for idx, layer in enumerate(self.backbone):
            if isinstance(layer, nn.Conv2d):
                # 卷积层计算量：2*C_in*C_out*K*K*H*W / 1e9（GOPS）
                C_in = layer.in_channels
                C_out = layer.out_channels
                K = layer.kernel_size[0]
                # 近似特征图尺寸（AlexNet/VGG16规律：每5层缩小1/2）
                H = 224 // (2 ** (idx // 5))
                W = H
                compute = 2 * C_in * C_out * K * K * H * W / 1e9
                self.layer_compute_cost[idx] = compute
            elif isinstance(layer, nn.Linear):
                # 全连接层计算量：2*in_features*out_features / 1e9（GOPS）
                compute = 2 * layer.in_features * layer.out_features / 1e9
                self.layer_compute_cost[idx] = compute
            else:
                # 激活/池化层计算量忽略（论文简化假设）
                self.layer_compute_cost[idx] = 0.01
        return self.layer_compute_cost