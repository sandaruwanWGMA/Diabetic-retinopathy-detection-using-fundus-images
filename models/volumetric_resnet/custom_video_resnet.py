import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import CustomSigmoidBlock

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from util.helpers.count_parameters import model_params


class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        # Load the pretrained R3D model
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Freeze all parameters in the model
        for param in model.parameters():
            param.requires_grad = False

        # Modify the first convolutional layer to accept 2 input channels
        old_conv = model.stem[0]
        self.first_conv = nn.Conv3d(
            2,  # Change here from 1 to 2 for two input channels
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Initialize the weights for the new two-channel input by copying the original weights
        # Repeat weights across the two channels or use a mean if appropriate
        self.first_conv.weight.data = old_conv.weight.data.mean(
            dim=1, keepdim=True
        ).repeat(1, 2, 1, 1, 1)

        # Copy the modified first conv layer back to the stem
        model.stem[0] = self.first_conv

        self.features = nn.Sequential(*list(model.children())[:-1])

        self.custom_sigmoid_block = CustomSigmoidBlock()

    def forward(self, x):
        # Forward pass through the modified features
        x = self.features(x)
        x = self.custom_sigmoid_block(x)
        return x
