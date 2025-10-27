import torch
import torch.nn as nn
from model.base_model import BaseMultiExitModel


class MultiExitAlexNet(BaseMultiExitModel):
    """
    多出口AlexNet（论文3.4节：4个早退出点，适配VEC场景）
    基于原始AlexNet改造，在卷积层后添加早退出分支
    """

    def __init__(self, num_classes=7, ac_min=0.8):
        super().__init__(num_classes=num_classes, ac_min=ac_min)
        self._build_backbone()  # 构建主干网络
        self._register_exit_layers()  # 注册4个早退出层（论文1-59）
        self.calculate_layer_compute_cost()  # 计算每层计算量

    def _build_backbone(self):
        """构建AlexNet主干网络（论文3.4节结构）"""
        self.backbone = nn.Sequential(
            # 卷积块1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # 0
            nn.ReLU(inplace=True),  # 1
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # 2
            nn.MaxPool2d(kernel_size=3, stride=2),  # 3（早退出挂钩点1）

            # 卷积块2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 4
            nn.ReLU(inplace=True),  # 5
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # 6
            nn.MaxPool2d(kernel_size=3, stride=2),  # 7（早退出挂钩点2）

            # 卷积块3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 8
            nn.ReLU(inplace=True),  # 9

            # 卷积块4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 10
            nn.ReLU(inplace=True),  # 11

            # 卷积块5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 12
            nn.ReLU(inplace=True),  # 13
            nn.MaxPool2d(kernel_size=3, stride=2),  # 14（早退出挂钩点3）

            # 全连接块
            nn.Flatten(),  # 15
            nn.Linear(256 * 6 * 6, 4096),  # 16
            nn.ReLU(inplace=True),  # 17
            nn.Dropout(p=0.5),  # 18（早退出挂钩点4）
            nn.Linear(4096, 4096),  # 19
            nn.ReLU(inplace=True),  # 20
            nn.Dropout(p=0.5),  # 21
            nn.Linear(4096, self.num_classes)  # 22（主出口）
        )

        # 早退出挂钩点（论文3.4节：4个早退出位置）
        self.exit_hook_points = [3, 7, 14, 18]

    def _register_exit_layers(self):
        """注册4个早退出层（适配各挂钩点特征尺寸，论文3.4节）"""
        # 早退出1（挂钩点3：(B,96,13,13)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(96, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes)
        ))

        # 早退出2（挂钩点7：(B,256,6,6)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        ))

        # 早退出3（挂钩点14：(B,256,6,6)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        ))

        # 早退出4（挂钩点18：(B,4096)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_classes)
        ))


if __name__ == "__main__":
    # 测试代码（论文实验参数）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiExitAlexNet(num_classes=7, ac_min=0.8).to(device)

    # 测试前向传播与早退出
    x = torch.randn(8, 3, 224, 224).to(device)  # (B,C,H,W)
    main_prob, exit_probs = model(x, return_exit_probs=True)
    print(f"AlexNet Main Exit shape: {main_prob.shape} (Paper 3.4节)")
    print(f"Number of early exits: {len(exit_probs)} (Paper 4 exits)")
    for i, prob in enumerate(exit_probs):
        print(f"Early Exit {i + 1} shape: {prob.shape}")

    # 测试早退出决策
    exit_idx, acc = model.get_early_exit_decision(exit_probs)
    print(f"Early Exit Index: {exit_idx}, Accuracy: {acc:.4f} (Threshold: 0.8)")

    # 测试模型划分（划分点=10，车端执行前10层）
    vehicle_model, edge_model = model.partition_model(partition_point=10)
    print(f"Vehicle model layers: {len(vehicle_model)}")
    print(f"Edge model layers: {len(edge_model)}")