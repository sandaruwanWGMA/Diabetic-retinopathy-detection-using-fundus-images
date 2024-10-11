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


class CustomUpsamplingSlicesBlock(nn.Module):
    def __init__(self):
        super(CustomUpsamplingSlicesBlock, self).__init__()

        # Initial convolutional layer
        self.conv_layer = nn.Conv3d(
            in_channels=512,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
        )

        # Transpose convolutional layer to increase spatial and temporal dimensions
        self.transpose_conv_layer = nn.ConvTranspose3d(
            in_channels=128, out_channels=64, kernel_size=(4, 4, 4), stride=2, padding=1
        )

        # Refining convolutional layer to refine features
        self.refining_conv_layer = nn.Conv3d(
            in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1
        )

        # Final convolutional layer to reduce the channel dimension to 1
        self.final_conv_layer = nn.Conv3d(
            in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=0
        )

    def forward(self, x):
        # Apply initial convolution
        x = self.conv_layer(x)
        # Perform trilinear upsampling (replacing the bilinear upsampling due to 5D tensor)
        x = F.interpolate(
            x, scale_factor=(1, 4, 4), mode="trilinear", align_corners=True
        )
        # Apply transpose convolution
        x = self.transpose_conv_layer(x)
        # Apply refining convolution
        x = self.refining_conv_layer(x)
        # Perform final upsampling to match the desired spatial and temporal dimensions
        x = F.interpolate(x, size=(150, 256, 256), mode="trilinear", align_corners=True)
        # Apply final convolution to adjust channels
        x = self.final_conv_layer(x)
        return x


class CustomSigmoidBlock(nn.Module):
    def __init__(self):
        super(CustomSigmoidBlock, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply the sigmoid function
        x = self.sigmoid(x)
        # Reduce to a single scalar by taking the mean
        x = torch.mean(x)
        x = (x >= 0.5).float()
        return x
