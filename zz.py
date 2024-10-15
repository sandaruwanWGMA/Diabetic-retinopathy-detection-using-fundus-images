import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys


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
from metrics.visualization import save_plots, save_metrics_plot

# Import Custom Dataset
from data.dataloader import MRIDataset
from data.data_handling import split_dataset


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
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

    # Creating dataset instances
    train_dataset = MRIDataset("./datasets/train_filenames.txt", limit=10)
    val_dataset = MRIDataset("./datasets/val_filenames.txt", limit=13)

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    # print(len(val_loader))

    # for i, data in enumerate(train_loader, 0):
    #     high_res_images = data[1]
    #     low_res_images = data[0]

    for i, data in enumerate(val_loader):
        print(len(data[0]))

    # for val_data in val_loader:
    #     high_res_images2, low_res_images2 = val_data[1], val_data[0]
    #     print(low_res_images2)


if __name__ == "__main__":
    try:
        # Set the environment variable
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
