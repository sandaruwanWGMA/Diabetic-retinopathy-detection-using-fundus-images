from .degradation_network import DegradationNetwork
from .CustomDeepLab import CustomDeepLab


def create_model(opt):
    if opt.model_type == "deeplab":
        return CustomDeepLab(
            in_channels=opt.in_channels,
            num_classes=opt.num_classes,
            freeze_backbone=opt.freeze_backbone,
            freeze_classifier=opt.freeze_classifier,
        )
    elif opt.model_type == "multi_gdn":
        return DegradationNetwork(image_size=opt.image_size)
    else:
        raise ValueError(f"Unknown model type: {opt.model_type}")
