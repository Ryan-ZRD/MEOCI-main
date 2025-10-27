import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset.data_augmentation import augment_image


class BDD100KDataset(Dataset):
    """
    Section 5.1: Dataset & Preprocessing
    -------------------------------------
    - Input size: 224×224
    - Normalization: [0,1]
    - Augmentation: rotation, crop, color jitter (Eq.16)
    - Label source: BDD100K official JSON (bdd100k_labels_release/bdd100k/labels/)
    """

    def __init__(self, config, split: str = "train"):
        self.data_root = config["data_root"]
        self.split = split
        self.img_size = tuple(config.get("img_size", (224, 224)))
        self.augment = bool(config.get("augment", split == "train"))
        self.cache_dir = os.path.join(self.data_root, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # 构建路径缓存文件（避免重复IO）
        self.cache_path = os.path.join(self.cache_dir, f"{split}_index.npy")
        if os.path.exists(self.cache_path):
            cache = np.load(self.cache_path, allow_pickle=True).item()
            self.img_paths, self.labels = cache["paths"], cache["labels"]
        else:
            self.img_paths = self._get_img_paths()
            self.labels = self._load_labels()
            np.save(self.cache_path, {"paths": self.img_paths, "labels": self.labels})

    # ----------------------------------------------------------
    # 图像路径加载
    # ----------------------------------------------------------
    def _get_img_paths(self):
        # 对应 data/bdd100k/images/100k/train, val, test
        img_dir = os.path.join(self.data_root, "images", "100k", self.split)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"[Dataset] Missing directory: {img_dir}")
        img_paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        if len(img_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")
        return sorted(img_paths)

    # ----------------------------------------------------------
    # 标签加载（支持 .txt 与官方 .json）
    # ----------------------------------------------------------
    def _load_labels(self):
        labels_dir = os.path.join(
            self.data_root, "bdd100k_labels_release", "bdd100k", "labels"
        )
        label_file_txt = os.path.join(labels_dir, f"{self.split}_labels.txt")
        label_file_json = os.path.join(labels_dir, f"bdd100k_labels_images_{self.split}.json")

        if os.path.exists(label_file_txt):
            # 本地转换后的标签
            with open(label_file_txt, "r") as f:
                label_dict = {
                    line.strip().split(",")[0]: int(line.strip().split(",")[1])
                    for line in f
                }
        elif os.path.exists(label_file_json):
            # 官方 JSON 标签
            with open(label_file_json, "r") as f:
                json_data = json.load(f)
                label_dict = {}
                for item in json_data:
                    img_name = os.path.basename(item["name"])
                    attr = item.get("attributes", {})
                    label_value = self._map_label_to_index(attr)
                    label_dict[img_name] = label_value
        else:
            raise FileNotFoundError(f"No valid label file found in {labels_dir}")

        labels = []
        for path in self.img_paths:
            fname = os.path.basename(path)
            if fname not in label_dict:
                raise KeyError(f"Missing label for image: {fname}")
            labels.append(label_dict[fname])
        return labels

    # ----------------------------------------------------------
    # 标签映射规则（论文分类任务：天气/时间/场景）
    # ----------------------------------------------------------
    def _map_label_to_index(self, attr: dict) -> int:
        mapping = {
            "daytime": 0,
            "night": 1,
            "dawn/dusk": 2,
            "rainy": 3,
            "snowy": 4,
            "overcast": 5,
            "undefined": 6,
        }
        scene = attr.get("timeofday", "undefined").lower()
        weather = attr.get("weather", "undefined").lower()
        if weather in mapping:
            return mapping[weather]
        return mapping.get(scene, 6)

    # ----------------------------------------------------------
    # 核心数据加载逻辑
    # ----------------------------------------------------------
    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)

        # Section 5.1 数据增强策略
        if self.augment and self.split == "train":
            img = augment_image(img)

        label = self.labels[idx]
        return {"image": img, "label": label, "path": img_path}

    def __len__(self):
        return len(self.img_paths)


# ==========================================================
# DataLoader 接口（统一从 config 读取）
# ==========================================================
def get_bdd100k_dataloader(config, split: str = "train"):
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    dataset = BDD100KDataset(config, split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


# ==========================================================
# 快速测试
# ==========================================================
if __name__ == "__main__":
    # 示例 config（实际项目从 config.py 导入）
    config = {
        "data_root": "data/bdd100k",  # ✅ 数据根路径
        "img_size": (224, 224),
        "batch_size": 4,
        "num_workers": 2,
        "augment": True,
    }

    dataloader = get_bdd100k_dataloader(config, split="train")
    for batch in dataloader:
        print(f"[Batch] Image shape: {batch['image'].shape}")
        print(f"[Batch] Label example: {batch['label'][:5]}")
        print(f"[Batch] First path: {batch['path'][0]}")
        break
