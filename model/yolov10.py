import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseMultiExitModel
from ultralytics import YOLO  # 基于Ultralytics YOLOv10封装（扩展目标检测任务）


class MultiExitYOLOv10(BaseMultiExitModel):
    """
    多出口YOLOv10（扩展目标检测任务：3个早退出点）
    基于YOLOv10改造，在颈部FPN层添加早退出检测头
    """

    def __init__(self, model_path="yolov10n.pt", ac_min=0.85):
        """
        :param model_path: YOLOv10预训练模型路径（n/s/m/l/x）
        :param ac_min: 最小推理精度阈值（基于mAP@0.5，扩展任务）
        """
        super().__init__(num_classes=80, ac_min=ac_min)  # COCO数据集80类
        self.base_model = YOLO(model_path).model  # 加载YOLOv10基础模型
        self._split_backbone_neck_head()  # 拆分主干、颈部、检测头
        self._register_exit_layers()  # 注册3个早退出检测头
        self.calculate_layer_compute_cost()  # 计算每层计算量

    def _split_backbone_neck_head(self):
        """拆分YOLOv10为骨干、颈部、检测头（扩展模型结构）"""
        # YOLOv10结构拆分（基于Ultralytics模型定义）
        self.backbone = self.base_model.model[:18]  # 主干网络（前18层）
        self.neck = self.base_model.model[18:23]  # 颈部FPN（5层）
        self.head = self.base_model.model[23:]  # 检测头（3层）

        # 整合为主干网络（适配base_model接口）
        self.backbone = nn.Sequential(
            *self.backbone,
            *self.neck,
            *self.head
        )

        # 早退出挂钩点（颈部FPN层）
        self.exit_hook_points = [18, 20, 22]  # 颈部3个特征层位置

    def _register_exit_layers(self):
        """注册3个早退出检测头（适配目标检测任务，扩展逻辑）"""
        # 早退出1（挂钩点18：(B,256,28,28)→检测头）
        self.register_exit_layer(nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3 * (self.num_classes + 5), kernel_size=1, padding=0)  # 3锚框：(x,y,w,h,conf)+classes
        ))

        # 早退出2（挂钩点20：(B,512,14,14)→检测头）
        self.register_exit_layer(nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3 * (self.num_classes + 5), kernel_size=1, padding=0)
        ))

        # 早退出3（挂钩点22：(B,1024,7,7)→检测头）
        self.register_exit_layer(nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3 * (self.num_classes + 5), kernel_size=1, padding=0)
        ))

    def forward(self, x, return_exit_preds=False):
        """
        目标检测前向传播（扩展逻辑，适配YOLOv10输出格式）
        :param x: 输入图像（(B,3,640,640)）
        :param return_exit_preds: 是否返回各早退出检测结果
        :return: 主检测结果 + （可选）各早退出检测结果
        """
        exit_preds = []
        feat = x

        # 主干网络前向，在挂钩点输出早退出检测结果
        for layer_idx, layer in enumerate(self.backbone):
            feat = layer(feat)
            if layer_idx in self.exit_hook_points:
                hook_idx = self.exit_hook_points.index(layer_idx)
                if hook_idx < len(self.exit_layers):
                    exit_pred = self.exit_layers[hook_idx](feat)
                    # 解码检测结果（转换为边界框坐标）
                    decoded_pred = self._decode_pred(exit_pred, img_size=x.shape[2])
                    exit_preds.append(decoded_pred)

        # 主检测结果（主干网络最后一层）
        main_pred = self._decode_pred(feat, img_size=x.shape[2])

        if return_exit_preds:
            return main_pred, exit_preds
        return main_pred

    def _decode_pred(self, pred, img_size):
        """
        解码YOLO检测结果（扩展逻辑：转换为x1,y1,x2,y2格式）
        :param pred: 网络输出（(B, 3*(C+5), H, W)）
        :param img_size: 输入图像尺寸
        :return: 解码结果（(B, N, 6)：x1,y1,x2,y2,conf,cls）
        """
        B, C, H, W = pred.shape
        num_anchors = 3
        num_classes = self.num_classes
        # reshape：(B, 3, H, W, C+5)
        pred = pred.view(B, num_anchors, num_classes + 5, H, W).permute(0, 1, 3, 4, 2).contiguous()

        # 网格坐标计算
        grid_x = torch.arange(W, device=pred.device).repeat(H, 1).view(1, 1, H, W).float()
        grid_y = torch.arange(H, device=pred.device).repeat(W, 1).t().view(1, 1, H, W).float()
        # YOLOv10默认小锚框
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], device=pred.device).view(1, 3, 1, 1, 2)

        # 解码x,y（归一化到[0,1]）
        pred[..., 0] = (torch.sigmoid(pred[..., 0]) + grid_x) / W
        pred[..., 1] = (torch.sigmoid(pred[..., 1]) + grid_y) / H
        # 解码w,h（归一化到[0,1]）
        pred[..., 2] = torch.exp(pred[..., 2]) * anchors[..., 0] / img_size
        pred[..., 3] = torch.exp(pred[..., 3]) * anchors[..., 1] / img_size
        # 置信度与类别概率（sigmoid激活）
        pred[..., 4] = torch.sigmoid(pred[..., 4])
        pred[..., 5:] = torch.sigmoid(pred[..., 5:])

        # 转换为绝对坐标（x1,y1,x2,y2）
        pred[..., 0] = pred[..., 0] * img_size - pred[..., 2] * img_size / 2  # x1
        pred[..., 1] = pred[..., 1] * img_size - pred[..., 3] * img_size / 2  # y1
        pred[..., 2] = pred[..., 0] + pred[..., 2] * img_size  # x2
        pred[..., 3] = pred[..., 1] + pred[..., 3] * img_size  # y2

        # 展平为（B, 3*H*W, 6）
        pred = pred.view(B, num_anchors * H * W, num_classes + 5)
        return pred

    def get_early_exit_decision(self, exit_preds, iou_threshold=0.5, labels=None):
        """
        目标检测早退出决策（扩展逻辑：基于mAP@0.5）
        :param exit_preds: 各早退出检测结果列表
        :param iou_threshold: mAP计算IOU阈值
        :param labels: 真实标签（用于计算mAP）
        :return: exit_idx（早退出索引）、mAP（对应精度）
        """
        from adp_d3qn.metrics import calculate_map
        if labels is None:
            raise ValueError("Labels are required for mAP calculation (detection task)")

        for idx, pred in enumerate(exit_preds):
            current_map = calculate_map(pred, labels, iou_threshold=iou_threshold)
            if current_map >= self.ac_min:
                return idx, current_map
        # 主出口mAP
        main_pred = self.forward(torch.randn(1, 3, 640, 640).to(next(self.parameters()).device))
        main_map = calculate_map(main_pred, labels, iou_threshold=iou_threshold)
        return -1, main_map


if __name__ == "__main__":
    # 测试代码（扩展目标检测任务）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiExitYOLOv10(model_path="yolov10n.pt", ac_min=0.85).to(device)

    # 测试前向传播
    x = torch.randn(2, 3, 640, 640).to(device)
    main_pred, exit_preds = model(x, return_exit_preds=True)
    print(f"YOLOv10 Main Exit shape: {main_pred.shape} (detection task)")
    print(f"Number of early exits: {len(exit_preds)} (3 exits)")

    # 测试早退出决策（模拟标签）
    mock_labels = torch.tensor([[[0, 100, 100, 200, 200]]], device=device)  # (B, N, 5)：cls,x1,y1,x2,y2
    exit_idx, map_score = model.get_early_exit_decision(exit_preds, labels=mock_labels)
    print(f"Early Exit Index: {exit_idx}, mAP@0.5: {map_score:.4f} (Threshold: 0.85)")