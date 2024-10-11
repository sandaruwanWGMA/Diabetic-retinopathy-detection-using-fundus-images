import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        # First convolution: Adjusting channel dimension from 63 to 150
        self.conv1 = nn.Conv3d(
            1, 150, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
        )
        self.bn1 = nn.BatchNorm3d(150)
        # Second convolution: Processing with the same number of channels
        self.conv2 = nn.Conv3d(
            150, 150, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(150)
        # Shortcut connection to match the channel dimensions
        self.shortcut = nn.Sequential(
            nn.Conv3d(
                1, 150, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
            ),
            nn.BatchNorm3d(150),
        )

    def forward(self, x):
        # Shortcut path
        shortcut = self.shortcut(x)
        # First convolution
        out = F.relu(self.bn1(self.conv1(x)))
        # Second convolution
        out = self.bn2(self.conv2(out))
        # Element-wise addition (residual connection)
        out += shortcut
        out = F.relu(out)
        return out


# Creating the model
model = ResidualBlock()
print(model)

# Example input tensor
input_tensor = torch.randn(1, 1, 63, 256, 256)
output_tensor = model(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)
