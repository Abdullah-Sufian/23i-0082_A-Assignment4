"""
Automated architecture generator: turns a sampled config (see search_space.py)
into a concrete PyTorch nn.Module.

Supported building blocks
  conv   : standard kxk conv -> BN -> act                         (residual if shapes match)
  dwsep  : depthwise kxk conv -> BN -> act -> pointwise 1x1 -> BN -> act  (MobileNetV1-style)
  mbconv : 1x1 expand -> depthwise kxk -> 1x1 linear projection   (MobileNetV2 inverted residual)
"""
import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
}


def act_layer(name):
    return ACTIVATIONS[name](inplace=True)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
                                  nn.BatchNorm2d(cout), act_layer(act))
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


class DWSepBlock(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False),  # depthwise
            nn.BatchNorm2d(cin), act_layer(act),
            nn.Conv2d(cin, cout, 1, bias=False),                              # pointwise
            nn.BatchNorm2d(cout), act_layer(act))
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


class MBConvBlock(nn.Module):
    """Inverted residual block built on a depthwise separable core."""

    def __init__(self, cin, cout, k, stride, act, expand):
        super().__init__()
        hid = cin * expand
        self.body = nn.Sequential(
            nn.Conv2d(cin, hid, 1, bias=False), nn.BatchNorm2d(hid), act_layer(act),
            nn.Conv2d(hid, hid, k, stride, k // 2, groups=hid, bias=False), nn.BatchNorm2d(hid), act_layer(act),
            nn.Conv2d(hid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


def make_block(kind, cin, cout, k, stride, act, expand):
    if kind == "conv":
        return ConvBlock(cin, cout, k, stride, act)
    if kind == "dwsep":
        return DWSepBlock(cin, cout, k, stride, act)
    if kind == "mbconv":
        return MBConvBlock(cin, cout, k, stride, act, expand)
    raise ValueError(f"unknown block type {kind}")


class GeneratedNet(nn.Module):
    """CNN generated from an architecture config."""

    def __init__(self, config, num_classes=10):
        super().__init__()
        self.config = config
        first_act = config["stages"][0]["activation"]
        cin = config["stem_channels"]
        self.stem = nn.Sequential(nn.Conv2d(3, cin, 3, 1, 1, bias=False), nn.BatchNorm2d(cin), act_layer(first_act))
        stages = []
        for si, st in enumerate(config["stages"]):
            blocks = []
            for bi in range(st["depth"]):
                stride = 2 if (bi == 0 and si > 0) else 1
                blocks.append(make_block(st["block"], cin, st["channels"], st["kernel"], stride,
                                         st["activation"], st["expand"]))
                cin = st["channels"]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        last_act = config["stages"][-1]["activation"]
        if config.get("head_channels", 0):
            self.head = nn.Sequential(nn.Conv2d(cin, config["head_channels"], 1, bias=False),
                                      nn.BatchNorm2d(config["head_channels"]), act_layer(last_act))
            cin = config["head_channels"]
        else:
            self.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(config.get("dropout", 0.0)), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(self.stages(self.stem(x)))
        return self.classifier(torch.flatten(self.pool(x), 1))


def build_from_config(config, num_classes=10):
    return GeneratedNet(config, num_classes)
