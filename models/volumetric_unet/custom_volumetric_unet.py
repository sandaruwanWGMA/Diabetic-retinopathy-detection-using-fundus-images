# import torch
# import torch.nn as nn
# from monai.networks.nets import UNet


# class UNetWithPrint(UNet):
#     def forward(self, x):
#         print(f"Input shape: {x.shape}")
#         for name, layer in self.model.named_children():
#             print(f"{name} input shape: {x.shape}")
#             x = layer(x)
#             print(f"{name} output shape: {x.shape}")
#         return x


# # Define the UNet model
# unet_model = UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm="batch",
# )

# # Creating a dummy input tensor of shape [1, 1, 128, 128, 128]
# input_tensor = torch.randn(1, 1, 256, 256, 256)

# model = UNetWithPrint(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm="batch",
# )

# # Forward pass through the model
# output = model(input_tensor)

# print(output.shape)


import torch
import torch.nn as nn
from monai.networks.nets import UNet

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import CustomResidualInput, DepthUpsampleNet


class UNetWithoutFirstLayer(UNet):
    def forward(self, x):
        # Skip the first layer by manually forwarding through the rest
        x = list(self.model.children())[1](x)  # skip the first layer
        for name, layer in list(self.model.named_children())[2:]:
            x = layer(x)
        return x


# Custom UNet class with the ResidualUnit replacing the default input layer
class CustomUNetWithResidual(nn.Module):
    def __init__(self):
        super(CustomUNetWithResidual, self).__init__()

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

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        # Pass through the ResidualUnit
        x = self.residual_unit(x)
        # Pass through the rest of the pre-trained UNet layers
        x = self.unet(x)
        # Pass through depthUpSamplingNet
        x = self.depthUpSamplingNet(x)

        return x


# Instantiate the custom model
model = CustomUNetWithResidual()

input_tensor = torch.randn(1, 1, 30, 256, 256)

# Forward pass through the modified model
output = model(input_tensor)

print(f"Final output shape: {output.shape}")
