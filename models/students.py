"""
Lightweight student networks, implemented from scratch and adapted to 32x32 inputs.

All three students expose the same interface as the teacher:
    forward(x)                         -> logits
    forward(x, return_features=True)   -> (logits, [f_mid (8x8), f_final (4x4)])
so the distillation framework can pair student and teacher feature maps at the
same spatial resolutions (8x8 and 4x4).

CIFAR adaptation: ImageNet versions down-sample 32x (224 -> 7). On 32x32 images
that would leave a 1x1 map, so early strides are reduced and every network
down-samples 8x in total (32 -> 4).
"""
import torch
import torch.nn as nn


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=nn.ReLU6):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act is not None:
            layers.append(act(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2 / EfficientNet-Lite MBConv block: expand 1x1 -> depthwise kxk -> linear project 1x1."""

    def __init__(self, cin, cout, stride, expand, k=3):
        super().__init__()
        hidden = int(round(cin * expand))
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(cin, hidden, 1))
        layers += [ConvBNAct(hidden, hidden, k, stride, groups=hidden),
                   nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


# ---------------------------------------------------------------------------
# MobileNetV2 (Sandler et al., 2018)
# ---------------------------------------------------------------------------
class MobileNetV2(nn.Module):
    # t (expansion), c (channels), n (repeats), s (stride). ImageNet strides were
    # (1,2,2,2,1,2,1) with a stride-2 stem; for CIFAR the stem and the first
    # stride-2 stage use stride 1 -> total down-sampling 8x.
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=100, width_mult=1.0, dropout=0.2):
        super().__init__()
        cin = _make_divisible(32 * width_mult)
        self.last_channel = _make_divisible(1280 * max(1.0, width_mult))
        self.stem = ConvBNAct(3, cin, 3, 1)
        blocks, self.mid_index = [], None
        for t, c, n, s in self.CFG:
            cout = _make_divisible(c * width_mult)
            for i in range(n):
                blocks.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
            if c == 96:  # last 8x8 stage -> mid hint
                self.mid_index = len(blocks) - 1
        self.blocks = nn.Sequential(*blocks)
        self.head = ConvBNAct(cin, self.last_channel, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.last_channel, num_classes))
        _init_weights(self)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        f_mid = None
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i == self.mid_index:
                f_mid = x
        f_final = self.head(x)
        logits = self.classifier(torch.flatten(self.pool(f_final), 1))
        return (logits, [f_mid, f_final]) if return_features else logits


# ---------------------------------------------------------------------------
# SqueezeNet 1.1 (Iandola et al., 2016)
# ---------------------------------------------------------------------------
class Fire(nn.Module):
    """Squeeze (1x1) -> expand (1x1 || 3x3) -> concat."""

    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1), nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(nn.Conv2d(squeeze, e1, 1), nn.BatchNorm2d(e1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(squeeze, e3, 3, padding=1), nn.BatchNorm2d(e3), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], 1)


class SqueezeNet(nn.Module):
    """SqueezeNet 1.1 for CIFAR.

    Changes vs. ImageNet version: 3x3 stride-1 first conv, BatchNorm after every
    conv (much more stable training), and no ReLU on the final logits (the
    original ReLU would forbid negative logits, which breaks softmax-temperature KD).
    """

    def __init__(self, num_classes=100, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(nn.MaxPool2d(3, 2, 1), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64))     # 16x16
        self.stage2 = nn.Sequential(nn.MaxPool2d(3, 2, 1), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128))  # 8x8
        self.stage3 = nn.Sequential(nn.MaxPool2d(3, 2, 1), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                                    Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))                       # 4x4
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1),
                                        nn.AdaptiveAvgPool2d(1))
        _init_weights(self)

    def forward(self, x, return_features=False):
        x = self.stage1(self.stem(x))
        f_mid = self.stage2(x)
        f_final = self.stage3(f_mid)
        logits = torch.flatten(self.classifier(f_final), 1)
        return (logits, [f_mid, f_final]) if return_features else logits


# ---------------------------------------------------------------------------
# EfficientNet-Lite0 (Google, 2020)
# ---------------------------------------------------------------------------
class EfficientNetLite0(nn.Module):
    """EfficientNet-B0 topology with the 'Lite' edge modifications:
    no squeeze-and-excitation, ReLU6 instead of Swish, fixed 32-ch stem and
    1280-ch head (not scaled). Compound scaling coefficients of Lite0 are 1.0/1.0.
    """
    # expand, kernel, channels, repeats, stride  (ImageNet strides 1,2,2,2,1,2,1;
    # for CIFAR the stem and the 24-ch stage use stride 1 -> 8x total)
    CFG = [(1, 3, 16, 1, 1), (6, 3, 24, 2, 1), (6, 5, 40, 2, 2), (6, 3, 80, 3, 2),
           (6, 5, 112, 3, 1), (6, 5, 192, 4, 2), (6, 3, 320, 1, 1)]

    def __init__(self, num_classes=100, dropout=0.2, drop_connect=0.0):
        super().__init__()
        self.stem = ConvBNAct(3, 32, 3, 1)
        cin, blocks, self.mid_index = 32, [], None
        for e, k, c, n, s in self.CFG:
            for i in range(n):
                blocks.append(InvertedResidual(cin, c, s if i == 0 else 1, e, k))
                cin = c
            if c == 112:  # last 8x8 stage -> mid hint
                self.mid_index = len(blocks) - 1
        self.blocks = nn.Sequential(*blocks)
        self.head = ConvBNAct(cin, 1280, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        _init_weights(self)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        f_mid = None
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i == self.mid_index:
                f_mid = x
        f_final = self.head(x)
        logits = self.classifier(torch.flatten(self.pool(f_final), 1))
        return (logits, [f_mid, f_final]) if return_features else logits


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
