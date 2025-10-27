import torch
import numpy as np
from model.base_model import BaseMultiExitModel


def calculate_avg_delay(total_delay: float, num_vehicles: int) -> float:
    """
    计算平均推理延迟（论文公式14：delay_avg = ∑∑delay^i(t)/(m×T)，简化单批次计算）
    :param total_delay: 单车辆单次推理总延迟（ms，含车端+传输+边缘延迟）
    :param num_vehicles: 车辆数量（论文1-147表格：5-30）
    :return: 平均推理延迟（ms）
    """
    # 论文逻辑：多车辆延迟平均（假设单批次任务数=车辆数）
    avg_delay = total_delay / num_vehicles
    return round(avg_delay, 2)


def calculate_task_completion_rate(avg_delay: float, delay_tolerate: float) -> float:
    """
    计算任务完成率（论文5.4节：在容忍延迟内完成的任务占比）
    :param avg_delay: 任务平均推理延迟（ms）
    :param delay_tolerate: 任务容忍延迟（ms，论文1-147表格：AlexNet25ms/VGG16250ms）
    :return: 任务完成率（0-1）
    """
    # 论文逻辑：延迟≤容忍延迟则任务完成
    completion_rate = 1.0 if avg_delay <= delay_tolerate else 0.0
    return round(completion_rate, 4)


def calculate_inference_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算推理精度（论文3.4节：正确推理样本数/总样本数，适配分类任务）
    :param probs: 模型输出概率（(B, num_classes)）
    :param labels: 真实标签（(B,)）
    :return: 推理精度（0-1）
    """
    # 论文逻辑：取概率最大的类别作为预测结果
    preds = torch.argmax(probs, dim=1)
    # 计算正确预测数
    correct = torch.sum(preds == labels).item()
    # 精度=正确数/总样本数
    accuracy = correct / len(labels)
    return round(accuracy, 4)


def calculate_early_exit_probability(model: BaseMultiExitModel, partition_point: int) -> list:
    """
    计算早退出概率（论文公式6：pr_i^early = ∏(1-pr_j)×pr_i，j=1到i）
    :param model: 多出口DNN模型（AlexNet/VGG16）
    :param partition_point: 模型划分点（确定车端可访问的早退出层）
    :return: 各早退出层概率列表（含主出口概率）
    """
    # 论文1-64：离线预定义出口优先级概率pr_i（0≤pr_i≤1，主出口pr=1.0）
    if isinstance(model, model.__class__.__name__ == "MultiExitAlexNet"):
        # AlexNet4个早退出点（论文1-59：Exit1-Exit4）
        pr = [0.1, 0.2, 0.3, 0.4, 1.0]  # 最后一个为主出口
    elif isinstance(model, model.__class__.__name__ == "MultiExitVGG16"):
        # VGG165个早退出点（论文1-59：Exit1-Exit5）
        pr = [0.05, 0.15, 0.25, 0.35, 0.45, 1.0]  # 最后一个为主出口
    else:
        raise ValueError("Only AlexNet/VGG16 supported (Paper Experiment 5.4)")

    # 确定车端可访问的早退出层数量（基于划分点与主干层比例）
    exit_count = len(model.exit_layers)
    accessible_exits = min(
        partition_point // (len(model.backbone) // exit_count),  # 划分点对应出口数
        exit_count
    )

    # 论文公式6计算各出口早退出概率
    early_exit_probs = []
    for i in range(accessible_exits):
        # 乘积项：∏(1-pr_j)，j=1到i-1
        product_term = np.prod([1 - pr[j] for j in range(i)])
        # 早退出概率：product_term × pr_i
        pr_early = product_term * pr[i]
        early_exit_probs.append(round(pr_early, 4))

    # 主出口概率（所有早退出层均不触发的概率）
    main_exit_prob = np.prod([1 - pr[j] for j in range(accessible_exits)]) * pr[-1]
    early_exit_probs.append(round(main_exit_prob, 4))

    return early_exit_probs


def calculate_accuracy_loss(original_acc: float, multi_exit_acc: float) -> float:
    """
    计算精度损失（论文5.4.2节：多出口模型相对原始模型的精度下降）
    :param original_acc: 原始单出口模型精度（0-1）
    :param multi_exit_acc: 多出口模型精度（0-1）
    :return: 精度损失（%，保留2位小数）
    """
    # 论文逻辑：精度损失=（原始精度-多出口精度）×100%
    acc_loss = (original_acc - multi_exit_acc) * 100
    return round(acc_loss, 2)


def log_experiment_metrics(epoch: int, avg_delay: float, completion_rate: float,
                           accuracy: float, early_exit_probs: list, model_name: str) -> None:
    """
    实验指标日志打印（论文5.4节实验结果输出格式）
    :param epoch: 训练轮次
    :param avg_delay: 平均推理延迟（ms）
    :param completion_rate: 任务完成率（0-1）
    :param accuracy: 推理精度（0-1）
    :param early_exit_probs: 早退出概率列表
    :param model_name: 模型名称（AlexNet/VGG16）
    """
    # 适配论文表格格式的日志输出
    print(f"=" * 80)
    print(f"Experiment Metrics (Epoch {epoch} | Model: {model_name}) - Paper 5.4")
    print(f"=" * 80)
    print(f"Average Inference Delay: {avg_delay} ms (Tolerate: {25 if model_name == 'AlexNet' else 250} ms)")
    print(f"Task Completion Rate: {completion_rate:.2%}")
    print(f"Inference Accuracy: {accuracy:.2%} (Threshold: 80.00%)")
    print(f"Early Exit Probabilities:")
    for i, prob in enumerate(early_exit_probs[:-1]):
        print(f"  Exit {i + 1}: {prob:.2%}")
    print(f"  Main Exit: {early_exit_probs[-1]:.2%}")
    print(f"=" * 80)


if __name__ == "__main__":
    # 测试代码（基于论文实验参数）
    from model.alexnet import MultiExitAlexNet
    from model.vgg16 import MultiExitVGG16
    import torch

    # 1. 初始化模型（论文参数）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alexnet = MultiExitAlexNet(num_classes=7, ac_min=0.8).to(device)
    vgg16 = MultiExitVGG16(num_classes=7, ac_min=0.8).to(device)

    # 2. 测试平均延迟计算（论文1-147参数：10辆车，单车辆延迟50ms）
    avg_delay = calculate_avg_delay(total_delay=50.0, num_vehicles=10)
    print(f"AlexNet Avg Delay: {avg_delay} ms (Paper 5.4.4预期值：~5 ms)")

    # 3. 测试任务完成率（AlexNet容忍延迟25ms）
    completion_rate = calculate_task_completion_rate(avg_delay=20.0, delay_tolerate=25.0)
    print(f"AlexNet Completion Rate: {completion_rate:.2%} (Paper 5.4.4预期值：100.00%)")

    # 4. 测试推理精度（模拟概率与标签）
    probs = torch.tensor([[0.1, 0.8, 0.05, 0.03, 0.01, 0.005, 0.005]], device=device)
    labels = torch.tensor([1], device=device)
    accuracy = calculate_inference_accuracy(probs, labels)
    print(f"Inference Accuracy: {accuracy:.2%} (Paper 5.4.2预期值：≥80.00%)")

    # 5. 测试早退出概率（AlexNet划分点=5）
    early_exit_probs = calculate_early_exit_probability(alexnet, partition_point=5)
    print(f"AlexNet Early Exit Probs: {early_exit_probs} (Paper 5.4.2预期值：[0.1, 0.18, 0.216, 0.2592, 0.2592])")

    # 6. 测试精度损失（原始精度95%，多出口精度93.8%）
    acc_loss = calculate_accuracy_loss(original_acc=0.95, multi_exit_acc=0.938)
    print(f"Accuracy Loss: {acc_loss}% (Paper 5.4.2预期值：~1.2%)")

    # 7. 测试日志打印
    log_experiment_metrics(
        epoch=500,
        avg_delay=20.5,
        completion_rate=0.98,
        accuracy=0.89,
        early_exit_probs=early_exit_probs,
        model_name="AlexNet"
    )