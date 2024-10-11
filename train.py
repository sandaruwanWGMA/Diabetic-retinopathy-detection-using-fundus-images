import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments"

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


def main():
    device = setup_device()

    # Initialize the generator and discriminator
    generator = CustomUNet().to(device)
    discriminator = CustomResnet().to(device)

    # Initialize Metric Tracker
    metrics = MetricTracker()

    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Losses
    criterion = GANLoss(gan_mode="lsgan").to(device)

    # Learning rate schedulers
    scheduler_G = get_scheduler(opt_G, {"lr_policy": "step", "lr_decay_iters": 10})
    scheduler_D = get_scheduler(opt_D, {"lr_policy": "step", "lr_decay_iters": 10})

    # Define the path to the base directory containing both Low-Res and High-Res directories
    base_dir = "./MRI Dataset"

    # Create the dataset and dataloader
    mri_dataset = MRIDataset(base_dir)
    dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=True)  # TEMPORARY

    train_dataset, val_dataset, _ = split_dataset(mri_dataset)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    num_epochs = 50
    for epoch in range(num_epochs):
        # Reset or initialize metrics for the epoch
        epoch_metrics = MetricTracker()

        for i, data in enumerate(dataloader, 0):
            high_res_images = data[1].to(device)
            low_res_images = data[0].to(device)

            # Generate fake images from low-res images
            fake_images = generator(low_res_images)

            # Prepare data for the discriminator
            real_input = torch.cat((high_res_images, high_res_images), dim=1)
            fake_input = torch.cat((fake_images.detach(), high_res_images), dim=1)

            # ===================
            # Update discriminator
            # ===================
            discriminator.zero_grad()
            print(f"epoch: {epoch}")
            print("Real Input: ", real_input.shape)
            random_tensor = torch.rand(1, 2, 150, 256, 256, device=device)
            real_pred = discriminator(random_tensor)
            # print("Real Pred: ", real_pred)
            # loss_D_real = criterion(torch.tensor(real_pred), True)
            # print("loss_D_real: ", loss_D_real)

            # fake_pred = discriminator(fake_input)
            # loss_D_fake = criterion(torch.tensor(fake_pred), False)
            # print("loss_D_fake: ", loss_D_fake)

            # loss_D = (loss_D_real + loss_D_fake) / 2
            # loss_D.backward()
            # opt_D.step()

            # =================
            # Update generator
            # =================
            # generator.zero_grad()

            # We calculate the loss based on the generator's fake output.
            # fake_input_G = torch.cat((fake_images, high_res_images), dim=1)
            # fake_pred_G = discriminator(fake_input_G)
            # loss_G = criterion(fake_pred_G, True)

            # loss_G.backward()
            # opt_G.step()

            # Calculate and record metrics
            # epoch_metrics.dices.append(calculate_dice(fake_images, high_res_images))
            # epoch_metrics.ious.append(calculate_iou(fake_images, high_res_images))

            # sensitivity, specificity = calculate_sensitivity_specificity(
            #     fake_images, high_res_images
            # )
            # epoch_metrics.sensitivities.append(sensitivity)
            # epoch_metrics.specificities.append(specificity)

            # Logging
            # if (i + 1) % 20 == 0:
            #     print(
            #         f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], "
            #         f"Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}"
            #     )

        # Validation loop
        # generator.eval()
        # discriminator.eval()
        # with torch.no_grad():
        #     for val_data in val_loader:
        #         high_res_images, low_res_images = val_data[1].to(device), val_data[
        #             0
        #         ].to(device)

        #         pred = generator(low_res_images)
        #         ssim_index, psnr_value = calculate_ssim_psnr(
        #             pred, high_res_images, data_range=1.0
        #         )
        #         epoch_metrics.ssims.append(ssim_index)
        #         epoch_metrics.psnrs.append(psnr_value)

    #     # Save plots of metrics
    #     save_plots(epoch_metrics.dices, "Dice Coefficient", epoch)
    #     save_plots(epoch_metrics.ious, "IOU", epoch)

    #     # Update learning rate
    #     scheduler_G.step()
    #     scheduler_D.step()

    # # Save models for later use
    # torch.save(generator.state_dict(), "generator.pth")
    # torch.save(discriminator.state_dict(), "discriminator.pth")


if __name__ == "__main__":
    main()
