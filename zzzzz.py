import torch
import torch.nn as nn

# Define the 3D Convolution Layer for downsampling
conv3d_layer = nn.Conv3d(
    in_channels=1,  # Input channels as specified in the input shape
    out_channels=1,  # Output channels remain the same
    kernel_size=(7, 1, 1),  # Kernel size for depth, height, and width
    stride=(2, 1, 1),  # Stride for depth, height, and width
    padding=(3, 0, 0),  # Padding to control output shape for height and width
)

# Create a sample input tensor with the shape [1, 1, 256, 256, 256]
input_tensor = torch.randn(1, 1, 256, 256, 256)

# Apply the 3D convolution layer to the input tensor
output_tensor = conv3d_layer(input_tensor)

# Define the second 3D Convolution Layer for further downsampling
conv3d_layer2 = nn.Conv3d(
    in_channels=1,  # Input channels as specified from previous layer output
    out_channels=1,  # Output channels remain the same
    kernel_size=(5, 1, 1),  # Kernel size for depth, height, and width
    stride=(2, 1, 1),  # Stride for depth to halve it
    padding=(2, 0, 0),  # Padding for depth to ensure precise downsampling
)

# Apply the second 3D convolution layer to the output of the first layer
output_tensor2 = conv3d_layer2(output_tensor)

# Define the third 3D Convolution Layer for further downsampling
conv3d_layer3 = nn.Conv3d(
    in_channels=1,  # Input channels from previous layer output
    out_channels=1,  # Output channels remain the same
    kernel_size=(3, 1, 1),  # Kernel size for depth, height, and width
    stride=(2, 1, 1),  # Stride for depth to halve it
    padding=(1, 0, 0),  # Padding to maintain spatial dimensions
)

# Apply the third 3D convolution layer to the output of the second layer
output_tensor3 = conv3d_layer3(output_tensor2)

# Print the output shape to verify after Layer 3
print("Output Shape after Layer 3:", output_tensor3.shape)

# Define the fourth 3D Convolution Layer for further downsampling
conv3d_layer4 = nn.Conv3d(
    in_channels=1,  # Input channels from previous layer output
    out_channels=1,  # Output channels remain the same
    kernel_size=(3, 1, 1),  # Kernel size for depth, height, and width
    stride=(2, 1, 1),  # Stride for depth to halve it
    padding=(1, 0, 0),  # Padding to maintain spatial dimensions
)

# Apply the fourth 3D convolution layer to the output of the third layer
output_tensor4 = conv3d_layer4(output_tensor3)

# Print the output shape to verify after Layer 4
print("Output Shape after Layer 4:", output_tensor4.shape)


# Define the first 3D Transposed Convolution Layer for upsampling
upsample_layer1 = nn.ConvTranspose3d(
    in_channels=1,  # Keep the same number of input channels
    out_channels=1,  # Keep the same number of output channels
    kernel_size=(4, 1, 1),  # Adjusted kernel size for depth upsampling
    stride=(2, 1, 1),  # Stride to double the depth dimension
    padding=(1, 0, 0),  # Padding adjusted to ensure exact output shape
)

# Apply the first upsampling layer to the input tensor
upsample_output1 = upsample_layer1(output_tensor4)

# Print the output shape to verify
print("Upsampled Output Shape after Layer 1:", upsample_output1.shape)

# Final Adjustment for 3D Transposed Convolution Layer
upsample_layer2 = nn.ConvTranspose3d(
    in_channels=1,  # Keep the same number of input channels
    out_channels=1,  # Keep the same number of output channels
    kernel_size=(3, 1, 1),  # Reduced kernel size for precise depth upsampling
    stride=(2, 1, 1),  # Stride to double the depth dimension
    padding=(1, 0, 0),  # Padding adjusted for exact doubling
)

# Apply the final adjusted upsampling layer to the output of the first upsampling layer
upsample_output2 = upsample_layer2(upsample_output1)

# Print the output shape to verify
print("Final Adjusted Upsampled Output Shape after Layer 2:", upsample_output2.shape)
