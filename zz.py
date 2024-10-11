import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms


# Import your model definitions
from models.volumetric_resnet.custom_video_resnet import CustomResnet
from models.volumetric_unet.custom_volumetric_unet import CustomUNet

# Import utility functions
from util.losses import GANLoss, cal_gradient_penalty
from util.schedulers import get_scheduler

from metrics.metrics import (
    MetricTracker,
    calculate_dice,
    calculate_iou,
    calculate_sensitivity_specificity,
    calculate_ssim_psnr,
)

from metrics.save_image_triplets import save_image_triplets
from metrics.visualization import save_plots

# Import Custom Dataset
from data.dataloader import MRIDataset
from data.data_handling import split_dataset


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = setup_device()

# Initialize the generator and discriminator
generator = CustomUNet()
discriminator = CustomResnet()

# Initialize Metric Tracker
metrics = MetricTracker()

# Optimizers
opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Losses
criterion = GANLoss(gan_mode="lsgan")

# Learning rate schedulers
scheduler_G = get_scheduler(opt_G, {"lr_policy": "step", "lr_decay_iters": 10})
scheduler_D = get_scheduler(opt_D, {"lr_policy": "step", "lr_decay_iters": 10})

# Define the path to the base directory containing both Low-Res and High-Res directories
base_dir = "./MRI Dataset"

# Create the dataset and dataloader
mri_dataset = MRIDataset(base_dir)
dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=True)  # TEMPORARY


for val_data in dataloader:
    print(val_data[1].shape)