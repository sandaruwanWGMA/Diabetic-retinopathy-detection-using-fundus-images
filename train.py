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
    train_dataset = MRIDataset("./datasets/train_filenames.txt")
    val_dataset = MRIDataset("./datasets/val_filenames.txt")

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    num_epochs = 50
    train_loss = []
    epoch_metrics_dices = []
    epoch_metrics_ious = []
    epoch_metrics_ssims = []
    epoch_metrics_psnrs = []
    for epoch in range(num_epochs):
        # Reset or initialize metrics for the epoch
        epoch_metrics = MetricTracker()
        training_loss_accum = 0
        dice = 0
        iou = 0

        for i, data in enumerate(train_loader, 0):
            high_res_images = data[1]
            low_res_images = data[0]

            # Generate fake images from low-res images
            fake_images = generator(low_res_images)

            # Prepare data for the discriminator
            real_input = torch.cat((high_res_images, high_res_images), dim=1)
            fake_input = torch.cat((fake_images.detach(), high_res_images), dim=1)

            # ===================
            # Update discriminator
            # ===================
            discriminator.zero_grad()
            real_pred = discriminator(real_input)
            loss_D_real = criterion(real_pred, True)
            fake_pred = discriminator(fake_input)
            loss_D_fake = criterion(fake_pred, False)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            opt_D.step()

            # =================
            # Update generator
            # =================
            generator.zero_grad()

            # We calculate the loss based on the generator's fake output.
            fake_input_G = torch.cat((fake_images, high_res_images), dim=1)
            fake_pred_G = discriminator(fake_input_G)
            loss_G = criterion(fake_pred_G, True)

            loss_G.backward()
            opt_G.step()

            # Calculate and record metrics
            dice += calculate_dice(fake_images, high_res_images)
            iou += calculate_iou(fake_images, high_res_images)

            sensitivity, specificity = calculate_sensitivity_specificity(
                fake_images, high_res_images
            )
            epoch_metrics.sensitivities.append(sensitivity)
            epoch_metrics.specificities.append(specificity)

            # Logging
            # if (i + 1) % 2 == 0:
            #     print(
            #         f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], "
            #         f"Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}"
            #     )
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], "
                f"Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}"
            )

            epoch_metrics.losses.append((loss_G.item() + loss_D.item()) / 2)
            training_loss_accum += loss_G.item()

        train_loss.append(training_loss_accum / len(train_loader))
        epoch_metrics_dices.append(dice / len(train_loader))
        epoch_metrics_ious.append(iou / len(train_loader))

        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()

    # Plotting and saving loss plots
    save_plots(epoch_metrics_dices, "Dice Coefficient", num_epochs=num_epochs)
    save_plots(epoch_metrics_ious, "IOU", num_epochs=num_epochs)

    save_metrics_plot(
        train_loss,  # Training losses
        "Loss vs No of Epoches",
        "Epoches",
        "Loss",
        num_epochs=num_epochs,
    )

    # Save models for later use
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")


if __name__ == "__main__":
    try:
        # Set the environment variable
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
