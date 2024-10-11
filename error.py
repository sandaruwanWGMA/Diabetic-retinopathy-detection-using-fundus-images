import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms


# Import your model definitions
from models.volumetric_resnet.custom_video_resnet import CustomResnet


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = setup_device()


discriminator = CustomResnet().to(device)
random_tensor = torch.rand(1, 2, 150, 256, 256, device=device)
real_pred = discriminator(random_tensor)
print("Real Pred 02: ", real_pred)
