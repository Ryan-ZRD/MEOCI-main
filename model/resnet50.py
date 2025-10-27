import torch
import torch.nn as nn
from model.base_model import BaseMultiExitModel


class Bottleneck(nn.Module):
    """ResNet50瓶颈块（1×1+3×3+1×1卷积，论文扩展模型）"""
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
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class MultiExitResNet50(BaseMultiExitModel):
    """
    多出口ResNet50（论文扩展模型：6个早退出点）
    基于ResNet50改造，在各残差块后添加早退出分支
    """

    def __init__(self, num_classes=7, ac_min=0.8):
        super().__init__(num_classes=num_classes, ac_min=ac_min)
        self.inplanes = 64
        self._build_backbone()  # 构建主干网络
        self._register_exit_layers()  # 注册6个早退出层
        self.calculate_layer_compute_cost()  # 计算每层计算量

    def _build_backbone(self):
        """构建ResNet50主干网络（扩展模型结构）"""
        # 初始卷积层
        init_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 0
            nn.BatchNorm2d(64),  # 1
            nn.ReLU(inplace=True),  # 2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3（早退出挂钩点1）
        )

        # 残差块组
        layer1 = self._make_layer(Bottleneck, 64, 3)  # 4-15（早退出挂钩点2）
        layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)  # 16-31（早退出挂钩点3）
        layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)  # 32-55（早退出挂钩点4）
        layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)  # 56-64（早退出挂钩点5）

        # 全连接层（主出口）
        fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 65（早退出挂钩点6）
            nn.Flatten(),  # 66
            nn.Linear(512 * Bottleneck.expansion, self.num_classes)  # 67（主出口）
        )

        # 整合主干网络
        self.backbone = nn.Sequential(
            *init_layers,
            *layer1,
            *layer2,
            *layer3,
            *layer4,
            *fc_layers
        )

        # 早退出挂钩点（6个位置）
        self.exit_hook_points = [3, 15, 31, 55, 64, 65]

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建残差块组"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def _register_exit_layers(self):
        """注册6个早退出层（适配各挂钩点特征尺寸）"""
        # 早退出1（挂钩点3：(B,64,56,56)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes)
        ))

        # 早退出2（挂钩点15：(B,256,56,56)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        ))

        # 早退出3（挂钩点31：(B,512,28,28)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        ))

        # 早退出4（挂钩点55：(B,1024,14,14)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_classes)
        ))

        # 早退出5（挂钩点64：(B,2048,7,7)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_classes)
        ))

        # 早退出6（挂钩点65：(B,2048)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.num_classes)
        ))


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiExitResNet50(num_classes=7, ac_min=0.8).to(device)

    # 测试前向传播
    x = torch.randn(8, 3, 224, 224).to(device)
    main_prob, exit_probs = model(x, return_exit_probs=True)
    print(f"ResNet50 Main Exit shape: {main_prob.shape}")
    print(f"Number of early exits: {len(exit_probs)} (6 exits)")

    # 测试早退出决策
    exit_idx, acc = model.get_early_exit_decision(exit_probs)
    print(f"Early Exit Index: {exit_idx}, Accuracy: {acc:.4f}")