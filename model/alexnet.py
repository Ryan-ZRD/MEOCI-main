import torch
import torch.nn as nn
from model.base_model import BaseMultiExitModel


class MultiExitAlexNet(BaseMultiExitModel):
    """
    多出口AlexNet（论文3.4节：4个早退出点，适配VEC场景）
    改进点：
      ✅ 支持 return_exit_probs 参数；
      ✅ 注册每层 compute_cost；
      ✅ 支持 register_exit_layer() 接口；
      ✅ 对齐 EdgeInference/VehicleInference 调用。
    """

    def __init__(self, num_classes=7, ac_min=0.8):
        super().__init__(num_classes=num_classes, ac_min=ac_min)
        self._build_backbone()
        self._register_exit_layers()
        self.calculate_layer_compute_cost()
        print(f"[Model Init] MultiExitAlexNet initialized with {len(self.exit_layers)} exits")

    def _build_backbone(self):
        """构建AlexNet主干网络（论文3.4节结构）"""
        self.backbone = nn.Sequential(
            # 卷积块1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Exit 1

            # 卷积块2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Exit 2

            # 卷积块3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 卷积块4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 卷积块5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Exit 3

            # 全连接部分
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Exit 4
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)  # Main exit
        )

        # 早退出挂钩点
        self.exit_hook_points = [3, 7, 14, 18]

    def _register_exit_layers(self):
        """注册4个早退出层（论文1-59表）"""
        exits = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(96, 256), nn.ReLU(inplace=True), nn.Linear(256, self.num_classes)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Linear(512, self.num_classes)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Linear(512, self.num_classes)
            ),
            nn.Sequential(
                nn.Linear(4096, 1024), nn.ReLU(inplace=True), nn.Linear(1024, self.num_classes)
            )
        ]
        for e in exits:
            self.register_exit_layer(e)

    def calculate_layer_compute_cost(self):
        """计算每层计算量，用于资源分配（论文Eq.15d）"""
        for i, layer in enumerate(self.backbone):
            if isinstance(layer, nn.Conv2d):
                in_c, out_c, k_h, k_w = layer.weight.shape
                layer.compute_cost = in_c * out_c * k_h * k_w / 1e6  # 简化为MFlops
            elif isinstance(layer, nn.Linear):
                layer.compute_cost = layer.in_features * layer.out_features / 1e6
            else:
                layer.compute_cost = 0.0

    def forward(self, x, return_exit_probs=False):
        """前向传播（支持早退出概率输出）"""
        exit_probs = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.exit_hook_points:
                exit_idx = self.exit_hook_points.index(i)
                prob = self.exit_layers[exit_idx](x)
                exit_probs.append(prob)
        if return_exit_probs:
            return x, exit_probs
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiExitAlexNet(num_classes=7, ac_min=0.8).to(device)
    x = torch.randn(8, 3, 224, 224).to(device)
    main_prob, exit_probs = model(x, return_exit_probs=True)
    print(f"Main Output: {main_prob.shape}, Exits: {len(exit_probs)}")
    for i, p in enumerate(exit_probs):
        print(f"Exit {i+1} -> {p.shape}")
