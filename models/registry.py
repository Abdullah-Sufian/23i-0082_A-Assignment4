"""Single place that maps an architecture name (+ optional NAS config) to a model."""
from models.students import EfficientNetLite0, MobileNetV2, SqueezeNet
from models.teacher import ResNet50CIFAR

STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]


def build_model(arch, num_classes=100, nas_config=None, pretrained=False):
    arch = arch.lower()
    if arch == "resnet50":
        return ResNet50CIFAR(num_classes, pretrained=pretrained)
    if arch == "mobilenetv2":
        return MobileNetV2(num_classes)
    if arch == "squeezenet":
        return SqueezeNet(num_classes)
    if arch == "efficientnet_lite0":
        return EfficientNetLite0(num_classes)
    if arch == "nas":
        from q2_nas.generator import build_from_config
        assert nas_config is not None, "NAS model needs its architecture config"
        return build_from_config(nas_config, num_classes)
    raise ValueError(f"unknown architecture: {arch}")


def model_from_checkpoint(ckpt):
    """Rebuild a model from a checkpoint dict produced by common.utils.save_checkpoint."""
    model = build_model(ckpt["arch"], ckpt["num_classes"], ckpt.get("nas_config"))
    model.load_state_dict(ckpt["model"])
    return model.eval()
