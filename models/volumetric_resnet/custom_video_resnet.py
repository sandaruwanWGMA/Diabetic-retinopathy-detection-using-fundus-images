import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import CustomUpsamplingSlicesBlock


class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        # Load the pretrained R3D model
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Freeze all parameters in the model
        for param in model.parameters():
            param.requires_grad = False

        # Modify the first convolutional layer to accept 1 input channel
        old_conv = model.stem[0]
        self.first_conv = nn.Conv3d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.first_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        # Copy the modified first conv layer back to the stem
        model.stem[0] = self.first_conv

        self.features = nn.Sequential(*list(model.children())[:-2])

        self.custom_upsampling_slices_block = CustomUpsamplingSlicesBlock()

    def forward(self, x):
        # Forward pass through the modified features
        x = self.features(x)
        x = self.custom_upsampling_slices_block(x)
        return x
