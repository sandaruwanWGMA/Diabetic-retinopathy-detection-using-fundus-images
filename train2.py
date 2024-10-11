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

# Import Custom Dataset
from data.dataloader import MRIDataset


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    device = setup_device()

    # Initialize the generator and discriminator
    generator = CustomUNet().to(device)
    discriminator = CustomResnet().to(device)

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
    dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"EPOCH NUMBER: {epoch}")
        for i, data in enumerate(dataloader, 0):
            print(f"Value of i: {i}")
            high_res_images = data[1].to(device)
            low_res_images = data[1].to(device)

            # Generate fake images from low-res images
            fake_images = generator(low_res_images)

            # Prepare data for the discriminator
            real_input = torch.cat((high_res_images, high_res_images), dim=1)
            fake_input = torch.cat((fake_images.detach(), high_res_images), dim=1)

            print("shape of real_input: ", real_input.shape)

            # ===================
            # Update discriminator
            # ===================
            # discriminator.zero_grad()
            # real_pred = discriminator(real_input)
            # loss_D_real = criterion(real_pred, True)

            # fake_pred = discriminator(fake_input)
            # loss_D_fake = criterion(fake_pred, False)

            # loss_D = (loss_D_real + loss_D_fake) / 2
            # loss_D.backward()
            # opt_D.step()

            # =================
            # Update generator
            # =================
    #         generator.zero_grad()

    #         # We calculate the loss based on the generator's fake output.
    #         fake_input_G = torch.cat((fake_images, high_res_images), dim=1)
    #         fake_pred_G = discriminator(fake_input_G)
    #         loss_G = criterion(fake_pred_G, True)

    #         loss_G.backward()
    #         opt_G.step()

    #         # Logging
    #         if (i + 1) % 100 == 0:
    #             print(
    #                 f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}"
    #             )

    #         # Update learning rate
    #         scheduler_G.step()
    #         scheduler_D.step()

    # # Save models for later use
    # torch.save(generator.state_dict(), "generator.pth")
    # torch.save(discriminator.state_dict(), "discriminator.pth")


if __name__ == "__main__":
    main()
