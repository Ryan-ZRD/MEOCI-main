"""
DNN模型模块（论文多出口模型定义，支持AlexNet/VGG16/ResNet50/YOLOv10的早退出与模型划分）
"""
from .base_model import BaseMultiExitModel
from .alexnet import MultiExitAlexNet
from .vgg16 import MultiExitVGG16
from .resnet50 import MultiExitResNet50
from .yolov10 import MultiExitYOLOv10

__all__ = [
    "BaseMultiExitModel", "MultiExitAlexNet", "MultiExitVGG16",
    "MultiExitResNet50", "MultiExitYOLOv10"
]