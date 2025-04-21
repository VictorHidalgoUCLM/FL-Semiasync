import torch
# all nn libraries nn.layer, convs and loss functions

import torch.nn as nn

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_channels , out_channels , stride ):
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