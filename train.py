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
from metrics.visualization import save_plots, save_loss_plot

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

    # Define the path to the base directory containing both Low-Res and High-Res directories
    base_dir = "./MRI Dataset"

    # Create the dataset and dataloader
    mri_dataset = MRIDataset(base_dir)
    # dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=True)  # TEMPORARY

    train_dataset, val_dataset, _ = split_dataset(mri_dataset)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    num_epochs = 1
    for epoch in range(num_epochs):
        # Reset or initialize metrics for the epoch
        epoch_metrics = MetricTracker()

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
            # print(f"epoch: {epoch}")
            # print("Real Input: ", real_input.shape)
            real_pred = discriminator(real_input)
            # print("Real Pred: ", real_pred)
            # print("Real Pred Shape: ", real_pred.shape)
            # print("Real Pred: ", real_pred)
            loss_D_real = criterion(real_pred, torch.ones_like(real_pred))
            # print("loss_D_real: ", loss_D_real)
            # print("loss_D_real: ", loss_D_real)

            fake_pred = discriminator(fake_input)
            # print("Fake Pred Shape: ", fake_pred.shape)
            # print("Fake Pred: ", fake_pred)
            loss_D_fake = criterion(
                torch.tensor(fake_pred), torch.zeros_like(fake_pred)
            )
            # print("loss_D_fake: ", loss_D_fake)

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

            print("Generator checkpoint 01")

            # Calculate and record metrics
            epoch_metrics.dices.append(calculate_dice(fake_images, high_res_images))
            epoch_metrics.ious.append(calculate_iou(fake_images, high_res_images))

            sensitivity, specificity = calculate_sensitivity_specificity(
                fake_images, high_res_images
            )
            epoch_metrics.sensitivities.append(sensitivity)
            epoch_metrics.specificities.append(specificity)

            # Logging
            # if (i + 1) % 20 == 0:
            #     print(
            #         f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], "
            #         f"Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}"
            #     )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], "
                f"Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}"
            )

            epoch_metrics.losses.append((loss_G.item() + loss_D.item()) / 2)

        # Validation loop
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            val_loss_accum = 0
            for val_data in val_loader:
                print("Checkpoint 02")
                high_res_images, low_res_images = val_data[1], val_data[0]

                pred = generator(low_res_images)
                ssim_index, psnr_value = calculate_ssim_psnr(
                    pred, high_res_images, data_range=1.0
                )
                epoch_metrics.ssims.append(ssim_index)
                epoch_metrics.psnrs.append(psnr_value)
                print("Checkpoint 03")
                temp = discriminator(fake_input_G)
                loss_G_val = criterion(
                    temp,
                    torch.ones_like(temp),
                )
                val_loss_accum += loss_G_val.item()

            epoch_metrics.losses.append(val_loss_accum / len(val_loader))

        print("Checkpoint 04")
        # Save plots of metrics
        print("epoch_metrics.dices: ", epoch_metrics.dices)
        print("epoch_metrics.ious: ", epoch_metrics.ious)
        save_plots(epoch_metrics.dices, "Dice Coefficient", epoch)
        print("Checkpoint 05")
        save_plots(epoch_metrics.ious, "IOU", epoch)
        print("Checkpoint 06")

        # Plotting and saving loss plots
        print("Training losses: ", epoch_metrics.losses[:-1])
        print("Validation losses: ", [epoch_metrics.losses[-1]])
        save_loss_plot(
            epoch_metrics.losses[:-1],  # Training losses
            [epoch_metrics.losses[-1]],  # Validation loss for the epoch
            "Loss",
            "Epoch",
            "Loss",
            epoch,
        )

        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()

    # Save models for later use
    # torch.save(generator.state_dict(), "generator.pth")
    # torch.save(discriminator.state_dict(), "discriminator.pth")


if __name__ == "__main__":
    try:
        # Set the environment variable
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
