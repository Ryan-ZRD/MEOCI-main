import torch
import torch.nn as nn
from model.base_model import BaseMultiExitModel


class MultiExitVGG16(BaseMultiExitModel):
    """
    多出口VGG16（论文§3.4：5个早退出点）
    优化点：
      ✅ forward 显式实现，确保各出口正确触发；
      ✅ 每层附加 compute_cost；
      ✅ 初始化打印 summary；
      ✅ 与 AlexNet / ResNet50 接口完全一致。
    """

    def __init__(self, num_classes=7, ac_min=0.8):
        super().__init__(num_classes=num_classes, ac_min=ac_min)
        self._build_backbone()
        self._register_exit_layers()
        self.calculate_layer_compute_cost()
        self.summary()

    def _build_backbone(self):
        """构建VGG16主干网络（卷积5块 + 全连接3层）"""
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Exit 1

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Exit 2

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Exit 3

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Exit 4

            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Exit 5

            # FC layers
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes)
        )

        # 与论文一致的出口挂钩点索引
        self.exit_hook_points = [4, 9, 16, 23, 30]

    def _register_exit_layers(self):
        """注册5个早退出层"""
        exits = [
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, self.num_classes)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.num_classes)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, self.num_classes)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.num_classes)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, self.num_classes))
        ]
        for e in exits:
            self.register_exit_layer(e)

    def forward(self, x, return_exit_probs=False):
        """显式前向传播以触发早退出"""
        exit_probs = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.exit_hook_points:
                idx = self.exit_hook_points.index(i)
                prob = self.exit_layers[idx](x)
                exit_probs.append(nn.functional.softmax(prob, dim=1))
        if return_exit_probs:
            return x, exit_probs
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiExitVGG16(num_classes=7, ac_min=0.8).to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    main_prob, exits = model(x, return_exit_probs=True)
    print(f"[Test] Main output: {main_prob.shape}, exits: {len(exits)}")
    for i, e in enumerate(exits):
        print(f"Exit {i+1}: {e.shape}")
