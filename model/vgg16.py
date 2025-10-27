import torch
import torch.nn as nn
from model.base_model import BaseMultiExitModel


class MultiExitVGG16(BaseMultiExitModel):
    """
    多出口VGG16（论文3.4节：5个早退出点，适配VEC场景）
    基于原始VGG16改造，在各卷积块后添加早退出分支
    """

    def __init__(self, num_classes=7, ac_min=0.8):
        super().__init__(num_classes=num_classes, ac_min=ac_min)
        self._build_backbone()  # 构建主干网络
        self._register_exit_layers()  # 注册5个早退出层（论文1-59）
        self.calculate_layer_compute_cost()  # 计算每层计算量

    def _build_backbone(self):
        """构建VGG16主干网络（论文3.4节结构）"""
        self.backbone = nn.Sequential(
            # 卷积块1（Conv2d*2 + MaxPool）
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 0
            nn.ReLU(inplace=True),  # 1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 2
            nn.ReLU(inplace=True),  # 3
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4（早退出挂钩点1）

            # 卷积块2（Conv2d*2 + MaxPool）
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 5
            nn.ReLU(inplace=True),  # 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 7
            nn.ReLU(inplace=True),  # 8
            nn.MaxPool2d(kernel_size=2, stride=2),  # 9（早退出挂钩点2）

            # 卷积块3（Conv2d*3 + MaxPool）
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 10
            nn.ReLU(inplace=True),  # 11
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 12
            nn.ReLU(inplace=True),  # 13
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 14
            nn.ReLU(inplace=True),  # 15
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16（早退出挂钩点3）

            # 卷积块4（Conv2d*3 + MaxPool）
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 17
            nn.ReLU(inplace=True),  # 18
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 19
            nn.ReLU(inplace=True),  # 20
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 21
            nn.ReLU(inplace=True),  # 22
            nn.MaxPool2d(kernel_size=2, stride=2),  # 23（早退出挂钩点4）

            # 卷积块5（Conv2d*3 + MaxPool）
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 24
            nn.ReLU(inplace=True),  # 25
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 26
            nn.ReLU(inplace=True),  # 27
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 28
            nn.ReLU(inplace=True),  # 29
            nn.MaxPool2d(kernel_size=2, stride=2),  # 30（早退出挂钩点5）

            # 全连接块
            nn.Flatten(),  # 31
            nn.Linear(512 * 7 * 7, 4096),  # 32
            nn.ReLU(inplace=True),  # 33
            nn.Dropout(p=0.5),  # 34
            nn.Linear(4096, 4096),  # 35
            nn.ReLU(inplace=True),  # 36
            nn.Dropout(p=0.5),  # 37
            nn.Linear(4096, self.num_classes)  # 38（主出口）
        )

        # 早退出挂钩点（论文3.4节：5个早退出位置）
        self.exit_hook_points = [4, 9, 16, 23, 30]

    def _register_exit_layers(self):
        """注册5个早退出层（适配各挂钩点特征尺寸，论文3.4节）"""
        # 早退出1（挂钩点4：(B,64,112,112)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes)
        ))

        # 早退出2（挂钩点9：(B,128,56,56)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes)
        ))

        # 早退出3（挂钩点16：(B,256,28,28)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        ))

        # 早退出4（挂钩点23：(B,512,14,14)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        ))

        # 早退出5（挂钩点30：(B,512,7,7)→(B,num_classes)）
        self.register_exit_layer(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_classes)
        ))


if __name__ == "__main__":
    # 测试代码（论文实验参数）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiExitVGG16(num_classes=7, ac_min=0.8).to(device)

    # 测试前向传播与早退出
    x = torch.randn(8, 3, 224, 224).to(device)
    main_prob, exit_probs = model(x, return_exit_probs=True)
    print(f"VGG16 Main Exit shape: {main_prob.shape} (Paper 3.4节)")
    print(f"Number of early exits: {len(exit_probs)} (Paper 5 exits)")

    # 测试模型划分（划分点=10，车端执行前10层）
    vehicle_model, edge_model = model.partition_model(partition_point=10)
    print(f"Vehicle model layers: {len(vehicle_model)}")
    print(f"Edge model layers: {len(edge_model)}")

    # 测试每层计算量
    layer_compute = model.layer_compute_cost
    print(f"Layer 0 (Conv2d) compute cost: {layer_compute[0]:.4f} GOPS")
    print(f"Layer 32 (Linear) compute cost: {layer_compute[32]:.4f} GOPS")