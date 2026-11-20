"""
Teacher network: ResNet-50 adapted to 32x32 CIFAR images.

Adaptation (standard for CIFAR): the ImageNet stem (7x7 stride-2 conv + max-pool,
which would shrink a 32x32 image to 8x8 immediately) is replaced by a 3x3 stride-1
conv and the max-pool is removed. The rest of torchvision's ResNet-50
(bottleneck blocks [3, 4, 6, 3]) is kept, so ImageNet weights can optionally be
loaded for every layer except the stem and classifier.

forward(x, return_features=True) also returns intermediate feature maps used by
feature-based distillation:
    layer3 -> (B, 1024, 8, 8)   "mid" hint
    layer4 -> (B, 2048, 4, 4)   "final" hint
"""
import torch
import torch.nn as nn
import torchvision


class ResNet50CIFAR(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(2048, num_classes)
        self.net = net

    def forward(self, x, return_features=False):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.layer1(x)
        x = n.layer2(x)
        f_mid = n.layer3(x)
        f_final = n.layer4(f_mid)
        logits = n.fc(torch.flatten(n.avgpool(f_final), 1))
        if return_features:
            return logits, [f_mid, f_final]
        return logits
