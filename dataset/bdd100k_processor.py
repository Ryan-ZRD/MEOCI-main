import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset.data_augmentation import augment_image

class BDD100KDataset(Dataset):
    """
    BDD100K数据集加载器（论文5.1节实验数据集）
    支持图像分类任务，适配AlexNet/VGG16输入要求
    """
    def __init__(self, data_root, split="train", img_size=(224, 224), augment=True):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size  # 论文中AlexNet/VGG16输入尺寸
        self.augment = augment
        self.img_paths = self._get_img_paths()
        self.labels = self._load_labels()

    def _get_img_paths(self):
        """获取图像路径（论文实验使用BDD100K的images目录）"""
        img_dir = os.path.join(self.data_root, "images", self.split)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"BDD100K {split} images not found at {img_dir} (Paper 5.1)")
        return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def _load_labels(self):
        """加载标签（论文分类任务：基于场景标签，如雨天、白天等）"""
        label_path = os.path.join(self.data_root, "labels", f"{self.split}_labels.txt")
        with open(label_path, "r") as f:
            label_dict = {line.strip().split(",")[0]: int(line.strip().split(",")[1]) for line in f}
        return [label_dict[os.path.basename(path)] for path in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """加载单样本：图像预处理+标签（论文5.1节数据压缩逻辑）"""
        # 图像读取与压缩（论文提及压缩图像以降低传输/计算延迟）
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB格式
        img = cv2.resize(img, self.img_size)  # 缩放至模型输入尺寸
        img = img / 255.0  # 归一化到[0,1]
        img = np.transpose(img, (2, 0, 1))  # (H,W,C)→(C,H,W)（PyTorch格式）

        # 数据增强（训练阶段启用，论文5.1节数据增强策略）
        if self.augment and self.split == "train":
            img = augment_image(img)

        # 标签获取
        label = self.labels[idx]

        return {
            "image": img.astype(np.float32),
            "label": label,
            "img_path": img_path
        }

def get_bdd100k_dataloader(data_root, split="train", batch_size=32, img_size=(224, 224), augment=True, num_workers=4):
    """
    获取BDD100K数据加载器（论文实验数据加载逻辑）
    :return: PyTorch DataLoader
    """
    dataset = BDD100KDataset(data_root, split, img_size, augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),  # 训练集打乱
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )

if __name__ == "__main__":
    # 测试代码（论文实验参数）
    dataloader = get_bdd100k_dataloader(
        data_root="/path/to/bdd100k",
        split="train",
        batch_size=8,
        img_size=(224, 224),
        augment=True
    )
    for batch in dataloader:
        print(f"Image shape: {batch['image'].shape} (Paper AlexNet/VGG16 input)")
        print(f"Label shape: {len(batch['label'])} (Batch labels)")
        print(f"First image path: {batch['img_path'][0]}")
        break