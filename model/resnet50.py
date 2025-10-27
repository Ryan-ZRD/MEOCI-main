import torch
import torch.nn as nn
from model.base_model import BaseMultiExitModel


class Bottleneck(nn.Module):
    """ResNet50瓶颈块（论文扩展模型：1×1→3×3→1×1卷积）"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class MultiExitResNet50(BaseMultiExitModel):
    """多出口ResNet50（论文扩展模型：6个早退出点）"""

    def __init__(self, num_classes=7, ac_min=0.8):
        super().__init__(num_classes=num_classes, ac_min=ac_min)
        self.inplanes = 64
        self._build_backbone()
        self._register_exit_layers()
        self.calculate_layer_compute_cost()
        self.summary()

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建残差块组"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _build_backbone(self):
        """构建ResNet50主干网络（扩展模型结构）"""
        init_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 0
            nn.BatchNorm2d(64),  # 1
            nn.ReLU(inplace=True),  # 2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3（早退出挂钩点1）
        )

        # 残差层
        layer1 = self._make_layer(Bottleneck, 64, 3)     # 早退出2
        layer2 = self._make_layer(Bottleneck, 128, 4, 2) # 早退出3
        layer3 = self._make_layer(Bottleneck, 256, 6, 2) # 早退出4
        layer4 = self._make_layer(Bottleneck, 512, 3, 2) # 早退出5

        fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 早退出6
            nn.Flatten(),
            nn.Linear(512 * Bottleneck.expansion, self.num_classes)
        )

        self.backbone = nn.Sequential(
            *init_layers, *layer1, *layer2, *layer3, *layer4, *fc_layers
        )
        self.exit_hook_points = [3, 15, 31, 55, 64, 65]

    def _register_exit_layers(self):
        """注册6个早退出层"""
        exits = [
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, self.num_classes)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, self.num_classes)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.num_classes)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, self.num_classes)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                          nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, self.num_classes)),
            nn.Sequential(nn.Flatten(),
                          nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, self.num_classes))
        ]
        for e in exits:
            self.register_exit_layer(e)

    def forward(self, x, return_exit_probs=False):
        """重写前向传播，确保出口层生效"""
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
    model = MultiExitResNet50(num_classes=7, ac_min=0.8).to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    out, exits = model(x, return_exit_probs=True)
    print(f"[Test] Main output: {out.shape}, exits: {len(exits)}")
    for i, e in enumerate(exits):
        print(f"Exit {i+1}: {e.shape}")
