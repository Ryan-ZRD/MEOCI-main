import torch
import torch.nn.functional as F
import numpy as np
from model.base_model import BaseMultiExitModel


# ------------------------------------------------------------
# Eq.(6): pr_i^early = (Π_{j=1}^{i-1}(1 - pr_j)) * pr_i
# ------------------------------------------------------------
def calculate_early_exit_probability(model: BaseMultiExitModel, partition_point: int, dataloader=None, device="cpu"):
    """
    动态计算多出口模型的早退出概率 (Eq.6)
    若提供 dataloader，则根据 softmax 输出置信度动态估算 pr_i；
    否则使用模型中缓存的统计均值。
    """
    model.eval()
    exit_layers = model.exit_layers
    num_exits = len(exit_layers)

    # ---- Step 1. 估算每个出口的平均置信度 pr_i ----
    if dataloader is not None:
        conf_list = torch.zeros(num_exits, device=device)
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                imgs = batch["image"].to(device)
                feat = model.forward_until(imgs, partition_point)  # 划分点前的特征
                for i, exit_layer in enumerate(exit_layers):
                    logits = exit_layer(feat)
                    probs = F.softmax(logits, dim=1)
                    conf = probs.max(dim=1)[0].mean()
                    conf_list[i] += conf
                count += 1
                if count >= 10:  # 仅采样前10个batch估计即可
                    break
        pr = (conf_list / count).cpu().numpy().tolist()
    else:
        # 若未提供数据，则使用模型保存的统计结果或经验均值
        pr = getattr(model, "exit_confidence", [0.2 + 0.1 * i for i in range(num_exits)])

    # ---- Step 2. 计算每个出口的早退出概率 Eq.(6) ----
    early_exit_probs = []
    for i in range(num_exits):
        prior = np.prod([1 - pr[j] for j in range(i)])
        pr_early = prior * pr[i]
        early_exit_probs.append(pr_early)

    # 主出口（所有早退都未触发）
    main_exit_prob = np.prod([1 - pr[j] for j in range(num_exits)])
    early_exit_probs.append(main_exit_prob)
    return early_exit_probs


# ------------------------------------------------------------
# Eq.(7): ac_i(t) ≥ ac_min  精度约束检查
# ------------------------------------------------------------
def validate_early_exit_accuracy(model: BaseMultiExitModel, dataloader, partition_point, config):
    """
    验证早退出出口的推理精度 (Eq.7)
    保证 ac_i(t) ≥ ac_min
    """
    model.eval()
    device = config["device"]
    ac_min = config["ac_min"]
    exit_accuracies = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            # 提取划分点前的特征
            feat = model.forward_until(imgs, partition_point)

            for exit_idx, exit_layer in enumerate(model.exit_layers):
                logits = exit_layer(feat)
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item()
                exit_accuracies.append({
                    "exit_idx": exit_idx,
                    "accuracy": acc,
                    "meets_threshold": acc >= ac_min
                })

            # 主出口验证
            final_pred = model(imgs).argmax(dim=1)
            final_acc = (final_pred == labels).float().mean().item()
            exit_accuracies.append({
                "exit_idx": "main",
                "accuracy": final_acc,
                "meets_threshold": final_acc >= ac_min
            })
            break  # 仅验证一批次即可
    return exit_accuracies


# ------------------------------------------------------------
# 测试示例
# ------------------------------------------------------------
if __name__ == "__main__":
    from model.vgg16 import MultiExitVGG16
    from dataset.bdd100k_processor import get_bdd100k_dataloader

    # === Config ===
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_root": "/path/to/bdd100k",
        "batch_size": 8,
        "img_size": (224, 224),
        "ac_min": 0.8,
    }

    # === Model & Data ===
    model = MultiExitVGG16(num_classes=10).to(config["device"])
    dataloader = get_bdd100k_dataloader(
        data_root=config["data_root"],
        split="val",
        batch_size=config["batch_size"],
        img_size=config["img_size"]
    )

    # === Example ===
    partition_point = 10
    early_exit_probs = calculate_early_exit_probability(model, partition_point, dataloader, device=config["device"])
    print("\n[Eq.6] Early Exit Probabilities:")
    for i, p in enumerate(early_exit_probs[:-1]):
        print(f"  Exit {i+1}: {p:.4f}")
    print(f"  Main Exit: {early_exit_probs[-1]:.4f}")

    acc_info = validate_early_exit_accuracy(model, dataloader, partition_point, config)
    print("\n[Eq.7] Early Exit Accuracy Check:")
    for info in acc_info:
        mark = "✓" if info["meets_threshold"] else "✗"
        print(f"  Exit {info['exit_idx']}: {info['accuracy']:.3f} (≥{config['ac_min']} {mark})")
