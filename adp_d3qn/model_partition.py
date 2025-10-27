import torch
import torch.nn as nn
from model.base_model import BaseMultiExitModel
from adp_d3qn.early_exit import calculate_early_exit_probability


def get_model_layer_compute_cost(model: BaseMultiExitModel):
    """
    计算DNN每层计算量（论文3.2节模型划分基础：c_j为第j层计算成本）
    :param model: 多出口模型（AlexNet/VGG16）
    :return: 每层计算量字典（key:层索引，value:计算量GOPS）
    """
    layer_compute = {}
    for idx, layer in enumerate(model.backbone.layers):
        if isinstance(layer, nn.Conv2d):
            # 卷积层计算量：2*C_in*C_out*K*K*H*W / 1e9（GOPS）
            C_in = layer.in_channels
            C_out = layer.out_channels
            K = layer.kernel_size[0]
            H = 224 // (2 ** (idx // 5))  # 近似特征图尺寸（AlexNet/VGG16规律）
            W = H
            compute = 2 * C_in * C_out * K * K * H * W / 1e9
            layer_compute[idx] = compute
        elif isinstance(layer, nn.Linear):
            # 全连接层计算量：2*in_features*out_features / 1e9（GOPS）
            compute = 2 * layer.in_features * layer.out_features / 1e9
            layer_compute[idx] = compute
        elif isinstance(layer, (nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d)):
            # 激活/池化/BN层计算量忽略（论文简化假设）
            layer_compute[idx] = 0.01
        else:
            layer_compute[idx] = 0.001
    return layer_compute


def get_edge_model_layers(backbone_layers, partition_point):
    """
    论文模型划分核心：获取边缘端需执行的层（划分点后所有层）
    :param backbone_layers: 模型主干层（nn.ModuleList）
    :param partition_point: 划分点par(t)
    :return: 边缘端层列表
    """
    if partition_point == 0:
        # 划分点0：全边缘执行（论文1-56：par(t)=0→车端不执行）
        return backbone_layers[:]
    elif partition_point == len(backbone_layers):
        # 划分点l：全车端执行（边缘端不执行）
        return []
    else:
        # 0<par<l：边缘端执行par后的层（论文1-56逻辑）
        return backbone_layers[partition_point:]


def decide_optimal_partition_point(agent, env_state, model: BaseMultiExitModel, layer_compute):
    """
    基于ADP-D3QN决策最优划分点（论文4.3节核心逻辑）
    :param agent: ADP-D3QN智能体
    :param env_state: 环境状态（ac(t), queue_e(t), c_e^rm(t), λ(t)）
    :param model: 多出口模型
    :param layer_compute: 每层计算量
    :return: 最优划分点par*
    """
    # 1. 生成所有可能的划分点（论文动作空间：par∈{0,1,...,l}）
    possible_partitions = list(range(len(model.backbone.layers) + 1))

    # 2. 预测每个划分点的Q值（ADP-D3QN评估）
    partition_q_values = []
    for par in possible_partitions:
        # 动作：(划分点par, 早退出点exit)（先固定exit为-1，后续联合优化）
        action = (par, -1)
        # 转换动作为索引（适配智能体动作空间）
        action_idx = par  # 简化：划分点直接作为动作索引（论文动作空间映射）
        # 预测Q值
        state_tensor = torch.tensor(env_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        q_value = agent.eval_net(state_tensor)[0, action_idx].item()
        partition_q_values.append(q_value)

    # 3. 选择Q值最大的划分点（论文优化目标：最小化延迟）
    optimal_par_idx = np.argmax(partition_q_values)
    optimal_par = possible_partitions[optimal_par_idx]

    # 4. 验证划分点可行性（满足论文约束：资源、延迟、精度）
    # 4.1 计算边缘端资源消耗（∑c_j ≤ C_e）
    edge_layers = get_edge_model_layers(model.backbone.layers, optimal_par)
    total_edge_compute = sum([layer_compute[layer_idx] for layer_idx in range(optimal_par, len(model.backbone.layers))])
    if total_edge_compute > agent.config["edge_resource"]:
        # 资源不足，选择次优划分点
        sorted_pars = [p for _, p in sorted(zip(partition_q_values, possible_partitions), reverse=True)]
        for par in sorted_pars[1:]:
            edge_compute = sum([layer_compute[layer_idx] for layer_idx in range(par, len(model.backbone.layers))])
            if edge_compute <= agent.config["edge_resource"]:
                optimal_par = par
                break

    # 4.2 计算早退出概率（确保精度满足ac_min，论文公式6）
    early_exit_prob = calculate_early_exit_probability(model, optimal_par)
    if early_exit_prob < agent.config["ac_min"]:
        # 精度不足，增加划分点（让边缘端处理更多层）
        optimal_par = min(optimal_par + 1, len(model.backbone.layers))

    return optimal_par


if __name__ == "__main__":
    # 测试：基于论文多出口AlexNet选择最优划分点
    from model.alexnet import MultiExitAlexNet
    from adp_d3qn.adp_d3qn_agent import ADPD3QNAgent
    from config import Config

    # 配置加载（论文参数）
    config = Config()
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.edge_resource = 15.0  # 边缘资源15GHz
    config.ac_min = 0.8  # 精度阈值

    # 模型与计算量加载
    model = MultiExitAlexNet(num_classes=10, ac_min=config.ac_min)
    layer_compute = get_model_layer_compute_cost(model)

    # 智能体初始化（ADP-D3QN）
    agent = ADPD3QNAgent(
        state_dim=4,  # 状态维度：ac, queue_e, c_e^rm, λ
        action_dim=len(model.backbone.layers) + 1,  # 动作数=划分点数量
        config=config.__dict__
    )

    # 模拟环境状态（论文1-103：s(t)=(ac(t), queue_e(t), c_e^rm(t), λ(t))）
    env_state = [0.85, 0.3, 0.7, 0.2]  # 精度0.85，队列占比0.3，剩余资源0.7，到达率0.2

    # 决策最优划分点
    optimal_par = decide_optimal_partition_point(agent, env_state, model, layer_compute)
    print(f"Optimal Partition Point for AlexNet (Paper): {optimal_par}")
    print(f"Edge Layers Count: {len(get_edge_model_layers(model.backbone.layers, optimal_par))}")
    print(
        f"Total Edge Compute Cost: {sum([layer_compute[i] for i in range(optimal_par, len(model.backbone.layers))]):.4f} GOPS")