import torch
import torch.nn as nn
from monai.networks.nets import UNet

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import CustomResidualInput, DepthUpsampleNet

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from util.helpers.count_parameters import model_params


class UNetWithoutFirstLayer(UNet):
    def forward(self, x):
        # Skip the first layer by manually forwarding through the rest
        x = list(self.model.children())[1](x)  # skip the first layer
        for name, layer in list(self.model.named_children())[2:]:
            x = layer(x)
        return x


# Custom UNet class with the ResidualUnit replacing the default input layer
class CustomUNet(nn.Module):
    def __init__(self):
        super(CustomUNet, self).__init__()

        # Add the custom ResidualUnit as the new input layer
        self.residual_unit = CustomResidualInput(in_channels=1, out_channels=16)

        # Load the pre-trained UNet, configured to skip its default input layer
        self.unet = UNetWithoutFirstLayer(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )

        self.depthUpSamplingNet = DepthUpsampleNet()

        # Freeze all layers in UNet model
        for param in self.unet.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Pass through the ResidualUnit
        x = self.residual_unit(x)
        # Pass through the rest of the pre-trained UNet layers
        x = self.unet(x)
        # Pass through depthUpSamplingNet
        x = self.depthUpSamplingNet(x)

        return x


print(model_params(CustomUNet()))
