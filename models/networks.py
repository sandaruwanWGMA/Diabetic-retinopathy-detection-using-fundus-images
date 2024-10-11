import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomResidualInput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomResidualInput, self).__init__()

        # Define the 3D Convolution Layer for downsampling
        self.residual_conv3d_layer = nn.Conv3d(
            in_channels=in_channels,  # Set based on input parameter
            out_channels=out_channels,  # Set based on input parameter
            kernel_size=(4, 4, 4),
            stride=(4, 2, 2),
            padding=(0, 1, 1),
        )

        # Define the 3D Transposed Convolution Layer for aligning depth
        self.align_depth_transposed_conv3d_layer = nn.ConvTranspose3d(
            in_channels=out_channels,  # Matches the previous layer's output channels
            out_channels=out_channels,  # Output channels remain the same
            kernel_size=(32, 1, 1),
            stride=(16, 1, 1),
            padding=(0, 0, 0),
        )

    def forward(self, x):
        # Apply the 3D convolution layer
        x = self.residual_conv3d_layer(x)

        # Apply the 3D transposed convolution layer
        x = self.align_depth_transposed_conv3d_layer(x)

        print("Output of Custom Input: ", x.shape)

        return x


class DepthUpsampleNet(nn.Module):
    def __init__(self):
        super(DepthUpsampleNet, self).__init__()
        # Layer 1: Transposed Convolution to adjust depth incrementally
        self.deconv1 = nn.ConvTranspose3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(4, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )

        # Layer 2: Another Transposed Convolution for fine adjustment
        self.deconv2 = nn.ConvTranspose3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(4, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )

    def forward(self, x):
        # Initial interpolation to near target using trilinear interpolation
        x = F.interpolate(
            x, size=(120, 256, 256), mode="trilinear", align_corners=False
        )

        # First transposed convolution
        x = self.deconv1(x)

        # Second transposed convolution to reach or approximate 150 depth
        x = self.deconv2(x)

        # Optional: Fine-tune the size if it's not exactly 150
        x = F.interpolate(
            x, size=(150, 256, 256), mode="trilinear", align_corners=False
        )

        return x
