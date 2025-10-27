import os
import json
import numpy as np
from tqdm import tqdm


def convert_bdd100k_labels(raw_label_dir, save_path, save_classmap=True):
    """
    将BDD100K原始JSON标签转换为分类任务标签（论文§5.1数据预处理）
    支持组合标签编码，如"clear+daytime"→class 0
    """
    # 组合标签定义（天气×时间，共7类）
    label_pairs = [
        ("clear", "daytime"), ("clear", "night"), ("rainy", "daytime"),
        ("rainy", "night"), ("foggy", "daytime"), ("snowy", "daytime"), ("dawn/dusk", "daytime")
    ]
    label_map = {f"{w}+{t}": i for i, (w, t) in enumerate(label_pairs)}
    if save_classmap:
        with open(os.path.join(os.path.dirname(save_path), "class_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)

    # 转换标签文件
    with open(save_path, "w") as label_file:
        for json_file in tqdm(os.listdir(raw_label_dir), desc="Converting BDD100K labels"):
            if not json_file.endswith(".json"):
                continue
            json_path = os.path.join(raw_label_dir, json_file)
            with open(json_path, "r") as f:
                data = json.load(f)
            attrs = data.get("attributes", {})
            weather = attrs.get("weather", "clear")
            timeofday = attrs.get("timeofday", "daytime")
            key = f"{weather}+{timeofday}"
            if key not in label_map:
                continue
            img_name = json_file.replace(".json", ".jpg")
            label = label_map[key]
            label_file.write(f"{img_name},{label}\n")

    print(f"[OK] Converted {len(os.listdir(raw_label_dir))} JSONs → {save_path}")
    print(f"[INFO] Class mapping saved to class_map.json (Paper §5.1)")


def calculate_data_statistics(dataloader):
    """
    计算数据集RGB均值与标准差（用于Normalization）
    """
    n_pixels = 0
    channel_sum = np.zeros(3)
    channel_squared_sum = np.zeros(3)

    for batch in tqdm(dataloader, desc="Computing statistics"):
        imgs = batch["image"].numpy()
        n_pixels += imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
        channel_sum += imgs.sum(axis=(0, 2, 3))
        channel_squared_sum += (imgs ** 2).sum(axis=(0, 2, 3))

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_squared_sum / n_pixels - mean ** 2)
    print(f"Dataset Mean (RGB): {mean.round(4)}, Std: {std.round(4)}")
    return mean, std


def check_data_integrity(data_root):
    """
    检查BDD100K数据完整性（Paper Dataset Verification）
    """
    required_dirs = ["images/train", "images/val", "labels"]
    for sub in required_dirs:
        full = os.path.join(data_root, sub)
        if not os.path.exists(full):
            return False, f"[ERROR] Missing directory: {full}"

    train_imgs = len(os.listdir(os.path.join(data_root, "images/train")))
    val_imgs = len(os.listdir(os.path.join(data_root, "images/val")))
    label_files = os.path.join(data_root, "labels", "train_labels.txt")

    if not os.path.exists(label_files):
        return False, f"[ERROR] Missing train_labels.txt under labels/"
    if train_imgs < 5000 or val_imgs < 1000:
        return False, f"[WARN] Insufficient samples (train={train_imgs}, val={val_imgs})"

    return True, f"[OK] Data verified (train={train_imgs}, val={val_imgs})"


if __name__ == "__main__":
    from dataset.bdd100k_processor import get_bdd100k_dataloader
    root = "/path/to/bdd100k"

    # 标签转换
    convert_bdd100k_labels(
        raw_label_dir=os.path.join(root, "labels/raw"),
        save_path=os.path.join(root, "labels/train_labels.txt")
    )

    # 数据统计
    dataloader = get_bdd100k_dataloader(data_root=root, split="train", batch_size=16)
    calculate_data_statistics(dataloader)

    # 数据完整性检查
    ok, msg = check_data_integrity(root)
    print(msg)
