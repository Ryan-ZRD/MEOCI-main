import torch
import torch.nn as nn
from ultralytics import YOLO
from model.base_model import BaseMultiExitModel


class MultiExitYOLOv10(BaseMultiExitModel):
    """
    多出口YOLOv10（扩展目标检测任务：3个早退出点）
    基于Ultralytics YOLOv10改造，论文§5.3中用于检测任务泛化实验。
    """

    def __init__(self, model_path="yolov10n.pt", ac_min=0.85):
        super().__init__(num_classes=80, ac_min=ac_min)
        self.base_model = YOLO(model_path).model
        self._split_backbone_neck_head()
        self._register_exit_layers()
        self.calculate_layer_compute_cost()
        self.summary()

    def _split_backbone_neck_head(self):
        """拆分YOLOv10为backbone + neck + head"""
        layers = list(self.base_model.model.children())
        if len(layers) < 25:  # 防止结构更新
            raise RuntimeError("YOLOv10层结构未对齐，请检查Ultralytics版本。")

        self.backbone = nn.Sequential(*layers[:18])
        self.neck = nn.Sequential(*layers[18:23])
        self.head = nn.Sequential(*layers[23:])
        self.backbone = nn.Sequential(*self.backbone, *self.neck, *self.head)

        # 论文设计：3个出口设置于FPN输出层
        self.exit_hook_points = [18, 20, 22]

    def _register_exit_layers(self):
        """注册3个早退出检测头"""
        def make_exit(in_ch, mid_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, 3 * (self.num_classes + 5), 1)
            )

        self.register_exit_layer(make_exit(256, 128))
        self.register_exit_layer(make_exit(512, 256))
        self.register_exit_layer(make_exit(1024, 512))

    def forward(self, x, return_exit_probs=False):
        """目标检测前向传播（与Base类接口保持一致）"""
        exit_preds = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in self.exit_hook_points:
                hook = self.exit_hook_points.index(idx)
                pred = self.exit_layers[hook](x)
                decoded = self._decode_pred(pred, img_size=x.shape[2])
                exit_preds.append(decoded)
        main_pred = self._decode_pred(x, img_size=x.shape[2])
        if return_exit_probs:
            return main_pred, exit_preds
        return main_pred

    def _decode_pred(self, pred, img_size):
        """YOLO输出解码为 (x1,y1,x2,y2,conf,cls)"""
        B, C, H, W = pred.shape
        num_a = 3
        pred = pred.view(B, num_a, self.num_classes + 5, H, W).permute(0, 1, 3, 4, 2).contiguous()
        gx = torch.arange(W, device=pred.device).repeat(H, 1).view(1, 1, H, W)
        gy = torch.arange(H, device=pred.device).repeat(W, 1).t().view(1, 1, H, W)
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], device=pred.device).view(1, 3, 1, 1, 2)

        pred[..., 0] = (torch.sigmoid(pred[..., 0]) + gx) / W
        pred[..., 1] = (torch.sigmoid(pred[..., 1]) + gy) / H
        pred[..., 2] = torch.exp(pred[..., 2]) * anchors[..., 0] / img_size
        pred[..., 3] = torch.exp(pred[..., 3]) * anchors[..., 1] / img_size
        pred[..., 4:] = torch.sigmoid(pred[..., 4:])

        pred[..., 0] = pred[..., 0] * img_size - pred[..., 2] * img_size / 2
        pred[..., 1] = pred[..., 1] * img_size - pred[..., 3] * img_size / 2
        pred[..., 2] = pred[..., 0] + pred[..., 2] * img_size
        pred[..., 3] = pred[..., 1] + pred[..., 3] * img_size
        return pred.view(B, num_a * H * W, self.num_classes + 5)

    def get_early_exit_decision(self, exit_preds, labels=None, iou_threshold=0.5):
        """检测任务早退出决策（基于mAP@0.5）"""
        from adp_d3qn.metrics import calculate_map
        if labels is None:
            raise ValueError("Labels required for detection evaluation.")
        for i, pred in enumerate(exit_preds):
            m = calculate_map(pred, labels, iou_threshold=iou_threshold)
            if m >= self.ac_min:
                return i, m
        m = calculate_map(exit_preds[-1], labels, iou_threshold=iou_threshold)
        return -1, m


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiExitYOLOv10("yolov10n.pt", ac_min=0.85).to(device)
    x = torch.randn(1, 3, 640, 640).to(device)
    main_pred, exits = model(x, return_exit_probs=True)
    print(f"Main output: {main_pred.shape}, exits: {len(exits)}")
