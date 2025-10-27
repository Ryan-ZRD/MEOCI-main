import os
import numpy as np
import json
from tqdm import tqdm


def convert_bdd100k_labels(raw_label_dir, save_path):
    """
    转换BDD100K原始标签为分类任务标签（论文5.1节标签处理逻辑）
    :param raw_label_dir: BDD100K原始标签目录（json格式）
    :param save_path: 保存路径（txt格式）
    """
    label_map = {
        "clear": 0, "rainy": 1, "snowy": 2, "foggy": 3,  # 天气场景（论文分类任务示例）
        "daytime": 4, "night": 5, "dawn/dusk": 6
    }
    label_file = open(save_path, "w")

    # 遍历原始标签文件
    for json_file in tqdm(os.listdir(raw_label_dir)):
        if not json_file.endswith(".json"):
            continue
        json_path = os.path.join(raw_label_dir, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        # 提取场景标签（论文关注环境感知相关场景）
        weather = data["attributes"]["weather"]
        timeofday = data["attributes"]["timeofday"]
        # 合并标签（如"clear+daytime"映射为0）
        if weather in label_map and timeofday in label_map:
            label = label_map[weather]
            img_name = json_file.replace(".json", ".jpg")
            label_file.write(f"{img_name},{label}\n")

    label_file.close()
    print(f"BDD100K labels converted to {save_path} (Paper 5.1)")


def calculate_data_statistics(dataloader):
    """
    计算数据集统计信息（均值、标准差，用于归一化）
    :param dataloader: 数据加载器
    :return: 各通道均值、标准差
    """
    mean = np.zeros(3)
    std = np.zeros(3)
    total_samples = 0

    for batch in tqdm(dataloader):
        imgs = batch["image"].numpy()  # (B,C,H,W)
        batch_size = imgs.shape[0]
        total_samples += batch_size

        # 计算每个通道的均值和标准差
        for c in range(3):
            mean[c] += imgs[:, c, :, :].mean() * batch_size
            std[c] += imgs[:, c, :, :].std() * batch_size

    # 全局均值和标准差
    mean /= total_samples
    std /= total_samples
    print(f"Dataset Mean (RGB): {mean.round(4)} (Paper normalization)")
    print(f"Dataset Std (RGB): {std.round(4)} (Paper normalization)")
    return mean, std


def check_data_integrity(data_root):
    """
    检查BDD100K数据集完整性（论文实验前数据校验）
    :param data_root: 数据集根路径
    :return: 完整性检查结果
    """
    required_dirs = ["images/train", "images/val", "labels"]
    for dir_path in required_dirs:
        full_path = os.path.join(data_root, dir_path)
        if not os.path.exists(full_path):
            return False, f"Missing directory: {full_path}"

    # 检查图像数量（论文实验使用约10K训练样本）
    train_imgs = len(os.listdir(os.path.join(data_root, "images/train")))
    val_imgs = len(os.listdir(os.path.join(data_root, "images/val")))
    if train_imgs < 5000 or val_imgs < 1000:
        return False, f"Insufficient samples: train={train_imgs}, val={val_imgs} (Paper requires ~10K train)"

    return True, f"Data integrity checked: train={train_imgs}, val={val_imgs} (Paper compliant)"


if __name__ == "__main__":
    # 测试标签转换（论文实验参数）
    convert_bdd100k_labels(
        raw_label_dir="/path/to/bdd100k/labels/raw",
        save_path="/path/to/bdd100k/labels/train_labels.txt"
    )

    # 测试数据统计
    from dataset.bdd100k_processor import get_bdd100k_dataloader

    dataloader = get_bdd100k_dataloader(
        data_root="/path/to/bdd100k",
        split="train",
        batch_size=32,
        augment=False
    )
    calculate_data_statistics(dataloader)

    # 测试数据完整性
    integrity, msg = check_data_integrity("/path/to/bdd100k")
    print(msg)