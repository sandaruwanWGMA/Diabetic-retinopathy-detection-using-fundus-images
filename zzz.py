# import torch
# import torch.nn as nn
# from monai.networks.nets import UNet


# # Subclass UNet to print input dimensions at each layer
# class UNetWithPrint(UNet):
#     def forward(self, x):
#         print(f"Input shape: {x.shape}")
#         for name, layer in self.model.named_children():
#             x = layer(x)
#             print(f"{name} output shape: {x.shape}")
#         return x


# # Instantiate the modified UNet model
# model = UNetWithPrint(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm="batch",
# )

# # Create a dummy input tensor of shape [1, 1, 256, 256, 256]
# input_tensor = torch.randn(1, 1, 256, 256, 256)

# # Forward pass through the modified model
# output = model(input_tensor)

# print(f"Final output shape: {output.shape}")


import torch
import torch.nn as nn
from monai.networks.nets import UNet


# Subclass UNet to print input dimensions at each layer
class UNetWithPrint(UNet):
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        for name, layer in self.model.named_children():
            x = layer(x)
            print(f"{name} output shape: {x.shape}")
        return x


# Subclass UNetWithPrint to skip the first layer
class UNetWithoutFirstLayer(UNet):
    def forward(self, x):
        # Skip the first layer by manually forwarding through the rest
        x = list(self.model.children())[1](x)  # skip the first layer
        for name, layer in list(self.model.named_children())[2:]:
            x = layer(x)
        return x


# Instantiate the modified UNet model without the first layer
model = UNetWithoutFirstLayer(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
)

# Create a dummy input tensor with the required shape
input_tensor = torch.randn(1, 16, 128, 128, 128)

print("Input Tensor: ", input_tensor.shape)

# Forward pass through the modified model
output = model(input_tensor)

print(f"Final output shape: {output.shape}")
