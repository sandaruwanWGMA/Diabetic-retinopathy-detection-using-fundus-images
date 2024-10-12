import torch
from torch.utils.data import DataLoader
from models.volumetric_unet.custom_volumetric_unet import CustomUNet
from models.volumetric_resnet.custom_video_resnet import CustomResnet
from data.dataloader import MRIDataset
from metrics.metrics import calculate_dice, calculate_iou, calculate_ssim_psnr
from util.losses import GANLoss  # Assuming this is the correct import for GANLoss


def load_model(model_path, model_class, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def validate_model(generator, discriminator, val_loader, device, criterion):
    # Ensure criterion is passed to function
    generator.eval()
    discriminator.eval()
    metrics = {"val_loss": [], "ssim_index": [], "psnr_value": []}
    with torch.no_grad():
        for val_data in val_loader:
            high_res_images, low_res_images = val_data[1].to(device), val_data[0].to(
                device
            )
            pred = generator(low_res_images)
            ssim_index_, psnr_value_ = calculate_ssim_psnr(
                pred, high_res_images, data_range=1.0
            )

            real_input_D = torch.cat((high_res_images, high_res_images), dim=1)
            fake_input_D = torch.cat((pred.detach(), high_res_images), dim=1)

            real_output_D = discriminator(real_input_D)
            fake_output_D = discriminator(fake_input_D)
            loss_D_real = criterion(real_output_D, torch.ones_like(real_output_D))
            loss_D_fake = criterion(fake_output_D, torch.zeros_like(fake_output_D))
            loss_D_val = (loss_D_real + loss_D_fake) / 2
            loss_G_val = criterion(
                discriminator(torch.cat((pred, high_res_images), dim=1)),
                torch.ones_like(real_output_D),
            )

            metrics["val_loss"].append(loss_G_val.item())
            metrics["ssim_index"].append(ssim_index_)
            metrics["psnr_value"].append(psnr_value_)

    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator_path = "generator.pth"
    discriminator_path = "discriminator.pth"

    # Load models
    generator = load_model(generator_path, CustomUNet, device)
    discriminator = load_model(discriminator_path, CustomResnet, device)

    # Prepare the loss function
    criterion = GANLoss(gan_mode="lsgan").to(device)

    # Load validation data
    val_dataset = MRIDataset("./datasets/val_filenames.txt")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Validate models
    validation_results = validate_model(
        generator, discriminator, val_loader, device, criterion
    )
    print("Validation Results:", validation_results)


if __name__ == "__main__":
    main()
