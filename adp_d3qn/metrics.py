import torch
import numpy as np
from model.base_model import BaseMultiExitModel


# ==========================================================
# 延迟相关
# ==========================================================
def calculate_avg_delay(total_delay_ms: float, num_vehicles: int) -> float:
    """
    Eq.(14): 平均推理延迟 delay_avg = ΣΣ delay_i(t) / (m×T)
    单批次简化：total_delay / num_vehicles
    """
    if num_vehicles <= 0:
        return 0.0
    avg_delay = total_delay_ms / num_vehicles
    return round(avg_delay, 3)


def calculate_task_completion_rate(avg_delay: float, delay_tolerate: float) -> float:
    """
    Eq.(15): 任务完成率 R_task = count(delay_i <= delay_tol) / total
    """
    completion_rate = 1.0 if avg_delay <= delay_tolerate else 0.0
    return round(completion_rate, 4)


# ==========================================================
# 精度与早退出相关
# ==========================================================
def calculate_inference_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Section 3.4: 推理精度 a_c = (#correct) / (#samples)
    """
    preds = torch.argmax(probs, dim=1)
    correct = torch.sum(preds == labels).item()
    accuracy = correct / len(labels)
    return round(accuracy, 4)


def calculate_accuracy_loss(original_acc: float, multi_exit_acc: float) -> float:
    """
    Section 5.4.2: 精度损失 ΔAcc = (Acc_baseline - Acc_ME) × 100%
    """
    return round((original_acc - multi_exit_acc) * 100, 2)


def calculate_early_exit_probability(model: BaseMultiExitModel, partition_point: int, config: dict) -> list:
    """
    Eq.(6): pr_i^early = ∏(1-pr_j) × pr_i
    使用模型/配置中定义的出口权重，支持动态出口数。
    """
    exit_count = len(getattr(model, "exit_layers", []))
    if exit_count == 0:
        return [1.0]

    # 从 config 动态读取或设定每层出口触发概率
    pr_config = config.get("exit_probabilities", None)
    if pr_config is None:
        # 自动生成 [0.1, 0.2, ... , 1.0]
        pr = [round(0.1 * (i + 1), 2) for i in range(exit_count - 1)] + [1.0]
    else:
        pr = list(pr_config) + [1.0] if pr_config[-1] < 1.0 else list(pr_config)

    # 确定可访问的出口层数量（按划分点映射）
    accessible_exits = min(
        partition_point // (len(model.backbone) // exit_count),
        exit_count
    )

    # 计算每层早退出概率
    early_exit_probs = []
    for i in range(accessible_exits):
        product_term = np.prod([1 - pr[j] for j in range(i)])
        pr_early = product_term * pr[i]
        early_exit_probs.append(round(pr_early, 4))

    # 主出口概率
    main_exit_prob = np.prod([1 - pr[j] for j in range(accessible_exits)]) * pr[-1]
    early_exit_probs.append(round(main_exit_prob, 4))
    return early_exit_probs


# ==========================================================
# 日志与展示
# ==========================================================
class MetricsLogger:
    """统一日志打印器（便于 main.py 调用）"""
    def __init__(self, config: dict):
        self.config = config

    def log(self, epoch: int, avg_delay: float, completion_rate: float,
            accuracy: float, early_exit_probs: list, model_name: str):
        delay_tol = self.config.get("delay_tolerate", 25.0 if "AlexNet" in model_name else 250.0)
        ac_min = self.config.get("ac_min", 0.8)

        print("=" * 90)
        print(f"[Epoch {epoch:04d}] Metrics Summary - Model: {model_name}")
        print("-" * 90)
        print(f"Average Inference Delay : {avg_delay:.3f} ms  (Tolerate: {delay_tol:.1f} ms)")
        print(f"Task Completion Rate    : {completion_rate * 100:.2f}%")
        print(f"Inference Accuracy      : {accuracy * 100:.2f}%  (Threshold: {ac_min * 100:.1f}%)")
        print("Early Exit Probabilities:")
        for i, p in enumerate(early_exit_probs[:-1]):
            print(f"  Exit-{i+1}: {p:.2%}")
        print(f"  Main Exit: {early_exit_probs[-1]:.2%}")
        print("=" * 90)

# ==========================================================
# 目标检测任务 mAP@0.5 计算（YOLOv10 扩展）
# ==========================================================
def calculate_map(preds: torch.Tensor, labels: torch.Tensor, iou_threshold: float = 0.5) -> float:
    """
    计算检测任务的 mAP@0.5（论文 §5.3）
    :param preds: 检测预测结果 (B, N, 85) -> [x1,y1,x2,y2,conf,cls...]
    :param labels: 真实框 (B, M, 5) -> [cls, x1, y1, x2, y2]
    :param iou_threshold: IoU 阈值（默认 0.5）
    :return: 平均 mAP 值
    """
    B = preds.shape[0]
    aps = []

    for b in range(B):
        pred = preds[b].detach().cpu().numpy()
        gt = labels[b].detach().cpu().numpy()

        if pred.size == 0 or gt.size == 0:
            aps.append(0.0)
            continue

        # 按置信度降序排序
        pred = pred[np.argsort(-pred[:, 4])]

        # 每个真实框是否匹配标志
        gt_matched = np.zeros(len(gt))
        tp, fp = [], []

        for p in pred:
            px1, py1, px2, py2, conf, cls = p[:6]
            # 只匹配同类
            same_cls = gt[gt[:, 0] == cls]
            if same_cls.shape[0] == 0:
                fp.append(1)
                tp.append(0)
                continue

            # 计算 IoU
            ix1 = np.maximum(px1, same_cls[:, 1])
            iy1 = np.maximum(py1, same_cls[:, 2])
            ix2 = np.minimum(px2, same_cls[:, 3])
            iy2 = np.minimum(py2, same_cls[:, 4])

            iw = np.maximum(ix2 - ix1, 0)
            ih = np.maximum(iy2 - iy1, 0)
            inter = iw * ih
            union = (px2 - px1) * (py2 - py1) + (same_cls[:, 3] - same_cls[:, 1]) * (same_cls[:, 4] - same_cls[:, 2]) - inter
            ious = inter / np.maximum(union, 1e-6)
            best_iou = np.max(ious)
            best_gt = np.argmax(ious)

            if best_iou > iou_threshold and not gt_matched[best_gt]:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt] = 1
            else:
                tp.append(0)
                fp.append(1)

        # 累积 TP/FP
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        precisions = tp / np.maximum(tp + fp, 1e-6)
        recalls = tp / len(gt)

        # 计算 AP（插值积分法）
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        i = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        aps.append(ap)

    return round(float(np.mean(aps)), 4)


# ==========================================================
# 测试示例
# ==========================================================
if __name__ == "__main__":
    from model.vgg16 import MultiExitVGG16

    config = {
        "ac_min": 0.8,
        "exit_probabilities": [0.05, 0.15, 0.25, 0.35, 0.45],
        "delay_tolerate": 250.0,
    }

    model = MultiExitVGG16(num_classes=10, ac_min=config["ac_min"])
    avg_delay = calculate_avg_delay(500, num_vehicles=25)
    completion_rate = calculate_task_completion_rate(avg_delay, config["delay_tolerate"])
    early_probs = calculate_early_exit_probability(model, partition_point=10, config=config)
    accuracy = 0.876
    acc_loss = calculate_accuracy_loss(0.91, accuracy)

    logger = MetricsLogger(config)
    logger.log(epoch=200, avg_delay=avg_delay, completion_rate=completion_rate,
               accuracy=accuracy, early_exit_probs=early_probs, model_name="VGG16")

    print(f"Accuracy Loss ΔAcc: {acc_loss:.2f}%")
