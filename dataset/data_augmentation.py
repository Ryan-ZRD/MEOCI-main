import numpy as np
import random

def augment_image(img):
    """
    图像数据增强（论文5.1节提及的随机翻转、亮度调整策略）
    :param img: 输入图像（np.ndarray, (C,H,W)）
    :return: 增强后图像
    """
    # 1. 随机水平翻转（50%概率，论文数据增强常用策略）
    if random.random() > 0.5:
        img = np.flip(img, axis=2)  # 沿宽度轴翻转

    # 2. 随机亮度调整（±10%，避免过曝/欠曝）
    brightness = random.uniform(0.9, 1.1)
    img = img * brightness
    img = np.clip(img, 0.0, 1.0)  # 裁剪到[0,1]范围

    # 3. 随机对比度调整（±10%，增强特征区分度）
    if random.random() > 0.5:
        mean = np.mean(img, axis=(1, 2), keepdims=True)  # 通道均值
        contrast = random.uniform(0.9, 1.1)
        img = (img - mean) * contrast + mean
        img = np.clip(img, 0.0, 1.0)

    return img

def resize_with_pad(img, target_size=(224, 224)):
    """
    等比例缩放+补边（避免图像畸变，适配目标检测任务扩展）
    :param img: 输入图像（np.ndarray, (H,W,C)）
    :param target_size: 目标尺寸（H,W）
    :return: 缩放后图像、缩放比例、补边信息
    """
    h, w = img.shape[:2]
    th, tw = target_size
    # 计算缩放比例（保持宽高比）
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    # 缩放图像
    img = cv2.resize(img, (nw, nh))
    # 计算补边（灰色填充，避免干扰特征）
    pad_left = (tw - nw) // 2
    pad_right = tw - nw - pad_left
    pad_top = (th - nh) // 2
    pad_bottom = th - nh - pad_top
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img, scale, (pad_left, pad_top)