import os
import nibabel as nib


def save_image_triplets(
    low_res_img, high_res_img, super_res_img, epoch, iteration, base_folder="results"
):
    # Define folder paths
    epoch_folder = os.path.join(base_folder, f"epoch_{epoch}")
    iteration_folder = os.path.join(epoch_folder, f"iter_{iteration}")

    # Create directories if they do not exist
    os.makedirs(os.path.join(iteration_folder, "low_res"), exist_ok=True)
    os.makedirs(os.path.join(iteration_folder, "high_res"), exist_ok=True)
    os.makedirs(os.path.join(iteration_folder, "super_res"), exist_ok=True)

    # File paths
    low_res_path = os.path.join(iteration_folder, "low_res", "low_res.nii")
    high_res_path = os.path.join(iteration_folder, "high_res", "high_res.nii")
    super_res_path = os.path.join(iteration_folder, "super_res", "super_res.nii")

    # Save .nii files
    nib.save(low_res_img, low_res_path)
    nib.save(high_res_img, high_res_path)
    nib.save(super_res_img, super_res_path)
