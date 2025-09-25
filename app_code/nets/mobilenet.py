import torch
# all nn libraries nn.layer, convs and loss functions

import torch.nn as nn

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HSwish(nn.Module):
    def forward(self, x):
        return x * torch.relu6(x + 3) / 6


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        reduced_channels = in_channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, activation):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        act_layer = nn.ReLU if activation == "RE" else HSwish

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(act_layer())

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_layer()
        ])

        if use_se:
            layers.append(SEBlock(hidden_dim))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class DepthWiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthWiseSeparable,self).__init__()
        
        # groups used here
        self.depthwise = nn.Conv2d(in_channels, in_channels, stride=stride, padding=1, kernel_size=3, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, stride=1, padding=0, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class MobileNetV1(nn.Module):
    
    def __init__(self, num_classes=10):
        
        super(MobileNetV1, self).__init__()

        # Initial convolution layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        
        # Depthwise separable convolutions
        self.features = nn.Sequential(
            self.features,
            DepthWiseSeparable(32, 64, 1),
            DepthWiseSeparable(64, 128, 2),
            DepthWiseSeparable(128, 128, 1),
            DepthWiseSeparable(128, 256, 2),
            DepthWiseSeparable(256, 256, 1),
            DepthWiseSeparable(256, 512, 2),
            
            DepthWiseSeparable(512, 512, 1),
            DepthWiseSeparable(512, 512, 1),
            DepthWiseSeparable(512, 512, 1),
            DepthWiseSeparable(512, 512, 1),
            DepthWiseSeparable(512, 512, 1),

            DepthWiseSeparable(512, 1024, 2),
            DepthWiseSeparable(1024, 1024, 1)

        )
        
        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MobileNetV3Small_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        act = HSwish()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),  # stride=1 para CIFAR-10
            nn.BatchNorm2d(16),
            act
        )

        self.blocks = nn.Sequential(
            InvertedResidual(16, 16, 3, 2, 1, True, "RE"),
            InvertedResidual(16, 24, 3, 2, 4.5, False, "RE"),
            InvertedResidual(24, 24, 3, 1, 3.67, False, "RE"),
            InvertedResidual(24, 40, 5, 2, 4.0, True, "HS"),
            InvertedResidual(40, 40, 5, 1, 6.0, True, "HS"),
        )

        self.final = nn.Sequential(
            nn.Conv2d(40, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            act,
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 1024, 1),
            act,
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        return x
    