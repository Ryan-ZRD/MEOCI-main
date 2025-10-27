import numpy as np
import torch
from model.base_model import BaseMultiExitModel


def calculate_early_exit_probability(model: BaseMultiExitModel, partition_point):
    """
    计算早退出概率（论文公式6：pr_i^early = ∏(1-pr_j) * pr_i，j=1到i）
    :param model: 多出口模型（AlexNet4个出口/VGG165个出口）
    :param partition_point: 划分点（确定车端可访问的出口层）
    :return: 各出口早退出概率列表
    """
    # 论文1-64：离线预计算的出口优先级概率pr_i（0≤pr_i≤1）
    if isinstance(model, model.__class__.__name__ == "MultiExitAlexNet"):
        # AlexNet4个早退出点（论文1-59：Exit1-Exit4）
        pr = [0.1, 0.2, 0.3, 0.4, 1.0]  # 最后一个为主出口（pr=1.0）
    elif isinstance(model, model.__class__.__name__ == "MultiExitVGG16"):
        # VGG165个早退出点（论文1-59：Exit1-Exit5）
        pr = [0.05, 0.15, 0.25, 0.35, 0.45, 1.0]  # 最后一个为主出口
    else:
        raise ValueError("Only AlexNet/VGG16 supported (Paper Experiment)")

    # 确定车端可访问的出口层（划分点前的出口）
    exit_count = len(model.exit_layers)
    accessible_exits = min(partition_point // (len(model.backbone.layers) // exit_count), exit_count)
    early_exit_probs = []

    # 论文公式6计算每个出口的早退出概率
    for i in range(accessible_exits):
        product_term = np.prod([1 - pr[j] for j in range(i)])  # ∏(1-pr_j)，j=1到i-1
        pr_early = product_term * pr[i]
        early_exit_probs.append(pr_early)

    # 主出口概率（所有早退出都不触发的概率）
    main_exit_prob = np.prod([1 - pr[j] for j in range(accessible_exits)]) * pr[-1]
    early_exit_probs.append(main_exit_prob)

    return early_exit_probs


def validate_early_exit_accuracy(model: BaseMultiExitModel, dataloader, partition_point, config):
    """
    验证早退出精度（论文5.4.2节：确保ac_i(t)≥ac_min）
    :param model: 多出口模型
    :param dataloader: 验证数据集（论文BDD100K）
    :param partition_point: 划分点
    :param config: 配置（含ac_min）
    :return: 各出口精度列表（是否满足阈值）
    """
    model.eval()
    device = config["device"]
    exit_accuracies = []
    ac_min = config["ac_min"]

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            # 车端前向（划分点前层）
            vehicle_model, _ = model.partition_model(partition_point)
            vehicle_feat = vehicle_model(imgs)

            # 计算各出口精度
            for exit_idx, exit_layer in enumerate(model.exit_layers):
                # 仅计算划分点可访问的出口
                if exit_idx > min(partition_point // (len(model.backbone.layers) // len(model.exit_layers)),
                                  len(model.exit_layers) - 1):
                    continue

                exit_prob = exit_layer(vehicle_feat)
                pred = torch.argmax(exit_prob, dim=1)
                accuracy = (pred == labels).float().mean().item()
                exit_accuracies.append({
                    "exit_idx": exit_idx,
                    "accuracy": accuracy,
                    "meets_threshold": accuracy >= ac_min
                })

            # 主出口精度
            main_prob = model(imgs)
            main_pred = torch.argmax(main_prob, dim=1)
            main_accuracy = (main_pred == labels).float().mean().item()
            exit_accuracies.append({
                "exit_idx": "main",
                "accuracy": main_accuracy,
                "meets_threshold": main_accuracy >= ac_min
            })
            break  # 简化：仅验证一个批次

    return exit_accuracies


if __name__ == "__main__":
    # 测试：计算VGG16早退出概率与精度
    from model.vgg16 import MultiExitVGG16
    from dataset.bdd100k_processor import get_bdd100k_dataloader
    from config import Config

    # 配置（论文参数）
    config = Config()
    config.data_root = "/path/to/bdd100k"
    config.batch_size = 8
    config.img_size = (224, 224)
    config.ac_min = 0.8

    # 模型与数据加载
    model = MultiExitVGG16(num_classes=10, ac_min=config.ac_min)
    dataloader = get_bdd100k_dataloader(
        data_root=config.data_root,
        split="val",
        batch_size=config.batch_size,
        img_size=config.img_size,
        augment=False
    )

    # 计算早退出概率（划分点=10，VGG16Block2后）
    partition_point = 10
    early_exit_probs = calculate_early_exit_probability(model, partition_point)
    print("VGG16 Early Exit Probabilities (Paper Eq.6):")
    for i, prob in enumerate(early_exit_probs[:-1]):
        print(f"Exit {i + 1}: {prob:.4f}")
    print(f"Main Exit: {early_exit_probs[-1]:.4f}")

    # 验证早退出精度
    exit_accuracies = validate_early_exit_accuracy(model, dataloader, partition_point, config)
    print("\nVGG16 Early Exit Accuracies (Paper 5.4.2):")
    for acc_info in exit_accuracies:
        status = "✓" if acc_info["meets_threshold"] else "✗"
        print(f"Exit {acc_info['exit_idx']}: {acc_info['accuracy']:.4f} (Threshold {config.ac_min}: {status})")