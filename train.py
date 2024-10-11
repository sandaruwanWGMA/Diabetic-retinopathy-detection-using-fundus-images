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


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank, world_size):
    setup(rank, world_size)
    device = setup_device()

    # Initialize the generator and discriminator
    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

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
    base_dir = "/kaggle/input/high-res-and-low-res-without-resample/Not Resampled"

    # Create the dataset and dataloader
    mri_dataset = MRIDataset(base_dir)

    train_dataset, val_dataset, _ = split_dataset(mri_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, sampler=train_sampler
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        val_dataset, batch_size=5, shuffle=False, sampler=val_sampler
    )

    num_epochs = 50
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        # Reset or initialize metrics for the epoch
        epoch_metrics = MetricTracker()

        for i, data in enumerate(train_loader, 0):
            print(f"Value of i: {i}")
            high_res_images = data[1].to(device)
            low_res_images = data[1].to(device)

            # ===================
            # Update discriminator
            # ===================
            discriminator.zero_grad()
            # Train with real MRI images
            real_pred = discriminator(high_res_images)
            loss_D_real = criterion(real_pred, True)
            # Train with fake MRI images
            fake_images = generator(low_res_images)
            fake_pred = discriminator(fake_images.detach())
            loss_D_fake = criterion(fake_pred, False)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            opt_D.step()

            # =================
            # Update generator
            # =================
            generator.zero_grad()
            fake_pred = discriminator(fake_images)
            loss_G = criterion(fake_pred, True)
            loss_G.backward()
            opt_G.step()

            # Calculate and record metrics
            pred = generator(low_res_images)
            epoch_metrics.dices.append(calculate_dice(pred, high_res_images))
            epoch_metrics.ious.append(calculate_iou(pred, high_res_images))
            sensitivity, specificity = calculate_sensitivity_specificity(
                pred, high_res_images
            )
            epoch_metrics.sensitivities.append(sensitivity)
            epoch_metrics.specificities.append(specificity)

            # Logging
            if (i + 1) % 20 == 0:
                print
                (
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}"
                )

            # Validation loop
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    high_res_images, low_res_images = data[1].to(device), data[0].to(
                        device
                    )

                    pred = generator(low_res_images)
                    ssim_index, psnr_value = calculate_ssim_psnr(
                        pred, high_res_images, data_range=1.0
                    )
                    epoch_metrics.ssims.append(ssim_index)
                    epoch_metrics.psnrs.append(psnr_value)

            if rank == 0:
                save_plots(metrics, "Dice Coefficient", epoch)
                save_plots(metrics, "IOU", epoch)

            # Update learning rate
            scheduler_G.step()
            scheduler_D.step()

    # Make sure only the master process saves the model
    if rank == 0:
        torch.save(generator.module.state_dict(), "generator.pth")
        torch.save(discriminator.module.state_dict(), "discriminator.pth")

    # Cleanup should be called after model saving
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
