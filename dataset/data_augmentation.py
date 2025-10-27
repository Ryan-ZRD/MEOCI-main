import cv2
import numpy as np
import random


def augment_image(img: np.ndarray) -> np.ndarray:
    """
    图像数据增强（论文 5.1 节）
    :param img: 输入图像 (np.ndarray, (C,H,W))
    :return: 增强后图像
    """
    # 1. 随机水平翻转 (50%)
    if random.random() < 0.5:
        img = np.flip(img, axis=2)

    # 2. 随机亮度 ±10%
    brightness = random.uniform(0.9, 1.1)
    img = np.clip(img * brightness, 0.0, 1.0)

    # 3. 随机对比度 ±10%
    if random.random() < 0.5:
        mean = np.mean(img, axis=(1, 2), keepdims=True)
        contrast = random.uniform(0.9, 1.1)
        img = np.clip((img - mean) * contrast + mean, 0.0, 1.0)

    # 4. 随机高斯噪声 (σ≈0.01)
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    # 5. 小角度旋转 (±5°)
    if random.random() < 0.3:
        c, h, w = img.shape
        center = (w // 2, h // 2)
        angle = random.uniform(-5, 5)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_cv = np.transpose(img, (1, 2, 0))  # C,H,W→H,W,C
        img_cv = cv2.warpAffine(img_cv, rot_mat, (w, h), borderValue=(0, 0, 0))
        img = np.transpose(img_cv, (2, 0, 1))

    return img.astype(np.float32)


def resize_with_pad(img: np.ndarray, target_size=(224, 224)):
    """
    等比例缩放+灰边填充 (论文 5.1 节扩展到检测任务)
    :param img: (H,W,C)
    :return: 缩放后图像、缩放比例、补边信息
    """
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    img = cv2.resize(img, (nw, nh))
    pad_left = (tw - nw) // 2
    pad_right = tw - nw - pad_left
    pad_top = (th - nh) // 2
    pad_bottom = th - nh - pad_top
    img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    return img, scale, (pad_left, pad_top)
