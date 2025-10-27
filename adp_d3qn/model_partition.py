import torch
import torch.nn as nn
import numpy as np
from model.base_model import BaseMultiExitModel
from adp_d3qn.early_exit import calculate_early_exit_probability


# -------------------------------------------------------------------------
# Eq.(5): 计算每层计算成本 c_j (GOPS)
# -------------------------------------------------------------------------
def get_model_layer_compute_cost(model: BaseMultiExitModel, input_resolution: int = 224):
    """
    计算每层的计算开销 c_j (GOPS)
    对应论文 Eq.(5) - "Computation Cost per Layer".
    动态根据层结构估算，不使用固定常数。
    """
    layer_compute = {}
    H = input_resolution
    W = input_resolution

    for idx, layer in enumerate(model.backbone):
        if isinstance(layer, nn.Conv2d):
            # 输出特征尺寸（动态计算）
            stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
            H_out = (H + 2 * padding - layer.kernel_size[0]) // stride + 1
            W_out = (W + 2 * padding - layer.kernel_size[0]) // stride + 1
            # Eq.(5): 2 * Cin * Cout * K^2 * H_out * W_out / 1e9
            compute = 2 * layer.in_channels * layer.out_channels * (layer.kernel_size[0] ** 2) * H_out * W_out / 1e9
            layer_compute[idx] = compute
            H, W = H_out, W_out  # 更新特征图尺寸
        elif isinstance(layer, nn.Linear):
            # 全连接层计算量 Eq.(5) 简化形式
            compute = 2 * layer.in_features * layer.out_features / 1e9
            layer_compute[idx] = compute
        elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            # 非线性层或BN层开销相对小，取平均每层常数项
            layer_compute[idx] = model.config.get("nonlinear_layer_cost", 0.005)
        else:
            # 其他层（dropout、flatten等）忽略
            layer_compute[idx] = 0.0

    return layer_compute


# -------------------------------------------------------------------------
# Section III-B: 划分点对应的边缘端层集合
# -------------------------------------------------------------------------
def get_edge_model_layers(backbone_layers, partition_point: int):
    """
    对应论文 Section III-B:
    The layers after partition point par(t) are executed at the edge.
    """
    if partition_point <= 0:
        return backbone_layers[:]  # 全部在边缘执行
    if partition_point >= len(backbone_layers):
        return []  # 全车端执行
    return backbone_layers[partition_point:]


# -------------------------------------------------------------------------
# Eq.(9–11): 基于ADP-D3QN的最优划分点选择
# -------------------------------------------------------------------------
def decide_optimal_partition_point(agent, env_state, model: BaseMultiExitModel, layer_compute: dict):
    """
    核心决策逻辑：
      - 通过ADP-D3QN计算每个划分点的Q值 (Eq.9)
      - 选择Q值最高的划分点 par*
      - 检查资源约束 (Eq.10) 与精度约束 (Eq.11)
    """
    device = agent.device
    state_tensor = torch.tensor(env_state, dtype=torch.float32).unsqueeze(0).to(device)

    num_layers = len(model.backbone)
    possible_partitions = list(range(num_layers + 1))
    q_values = []

    # === Step 1: 计算每个划分点的Q值 ===
    for par in possible_partitions:
        # 将划分点映射为动作索引（动作空间=划分点+出口点联合）
        # 这里只对划分点维度采样（exit=-1）
        action_idx = par
        q_value = agent.q_net(state_tensor)[0, action_idx].item()
        q_values.append(q_value)

    # === Step 2: 找到最大Q对应的划分点 par* ===
    optimal_par = possible_partitions[int(np.argmax(q_values))]

    # === Step 3: 检查边缘资源约束 Eq.(10) ===
    edge_layers = get_edge_model_layers(model.backbone, optimal_par)
    total_edge_compute = float(np.sum([layer_compute.get(i, 0) for i in range(optimal_par, num_layers)]))

    edge_capacity = agent.config.get("edge_resource", 15.0)
    if total_edge_compute > edge_capacity:
        # 若资源不足，从高到低依次尝试可行解
        feasible_par = None
        for par, q in sorted(zip(possible_partitions, q_values), key=lambda x: x[1], reverse=True):
            edge_compute = np.sum([layer_compute.get(i, 0) for i in range(par, num_layers)])
            if edge_compute <= edge_capacity:
                feasible_par = par
                break
        optimal_par = feasible_par if feasible_par is not None else num_layers  # 若全不满足则全车端执行

    # === Step 4: 检查精度约束 Eq.(11): ac(par) ≥ ac_min ===
    ac_min = agent.config.get("ac_min", 0.8)
    early_exit_prob = calculate_early_exit_probability(model, optimal_par)
    if early_exit_prob < ac_min:
        # 若当前划分精度不足，增大划分深度（让边缘执行更多层）
        optimal_par = min(optimal_par + 1, num_layers)

    return optimal_par


# -------------------------------------------------------------------------
# 简单测试 (可在论文附录中复现)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from model.alexnet import MultiExitAlexNet
    from adp_d3qn.adp_d3qn_agent import ADPD3QNAgent

    # === 参数加载（全部来自 config） ===
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "edge_resource": 15.0,     # Ce (GHz)
        "ac_min": 0.8,             # 最小精度阈值
        "nonlinear_layer_cost": 0.005
    }

    # === 初始化模型 & 智能体 ===
    model = MultiExitAlexNet(num_classes=10)
    layer_compute = get_model_layer_compute_cost(model, input_resolution=224)
    agent = ADPD3QNAgent(state_dim=4, action_dim=len(model.backbone) + 1, config=config)

    # === 模拟环境状态 s(t) ===
    env_state = [0.85, 0.3, 0.7, 0.2]

    # === 决策最优划分点 ===
    par = decide_optimal_partition_point(agent, env_state, model, layer_compute)
    print(f"[Test] Optimal partition point = {par}")
    print(f"[Test] Edge compute cost = {np.sum([layer_compute[i] for i in range(par, len(model.backbone))]):.4f} GOPS")
