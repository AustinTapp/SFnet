import os
import json
import time
import shutil
import tempfile
import resource
import sys
import pdb
import gc

import glob
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from monai import transforms
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from monai.losses import SSIMLoss
from monai.networks.nets import SwinUNETR
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from torch.nn import L1Loss, MSELoss
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator

obj = None
gc.collect()
torch.cuda.empty_cache()
print_config()
plt.close()

#seed
set_determinism(42)

# needed for WSL2
torch.multiprocessing.set_sharing_strategy('file_system')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

root_dir = str(os.getcwd())
LF = sorted(glob.glob(root_dir + "/Data/LF/*.nii.gz"))
HF = sorted(glob.glob(root_dir + "/Data/HF/*.nii.gz"))
print("Working directory is:", os.getcwd())

# Define Hyper-paramters for training loop
batch_size = 1
max_epochs = 2000
val_interval = 200
autoencoder_warm_up_n_epochs = 10
lr = 1e-4
warmup = 200

adv_weight = 0.25
perceptual_weight = 2

vit_model_path = None #"ssl_pretrained_weights.pth"
swin_pt_path = None #"ssl_pretrained_weights.pth" # pt weights monai
swin_model_path = None #"best_model.pt" # your previous runs

#image is the LF, label is the HF
image_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.GaussianSmoothd(keys=["label"], sigma=0.25, allow_missing_keys=True),
        transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(192, 192, 192)),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(96, 96, 96), random_size=False),
        transforms.ScaleIntensityRangePercentilesd(keys=["image", "label"], lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

train_data = []
for idx, path in enumerate(LF):
    image_path = LF
    label_path = HF

    data_point = {"image": image_path[idx], "label": label_path[idx]}
    train_data.append(data_point)

num_images = len([data["image"] for data in train_data])
label_images = len([data["label"] for data in train_data])

print("Size of 'image' list:", num_images)
print("Size of 'label' list:", label_images)

train_proportion = 0.92
train_size = int(train_proportion * num_images)
validation_size = num_images - train_size

print("Size of training dataset:", train_size)
print("Size of validation dataset:", validation_size)

train_subset, validation_subset = train_data[:train_size], train_data[train_size:]
train_ds = Dataset(data=train_subset, transform=image_transforms)
validation_ds = Dataset(data=validation_subset, transform=image_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, drop_last=True, persistent_workers=False, shuffle=True)

val_loader = DataLoader(validation_ds, batch_size=batch_size, drop_last=True, persistent_workers=False, shuffle=True)


if __name__ == "__main__":
    check_data = first(train_loader)

    image1 = check_data["image"]
    label1 = check_data["label"]

    print(f"image shape: {image1.shape}", f"label shape: {label1.shape}")

    plt.clf()
    im1 = image1[0, 0].detach().cpu().numpy()
    gt1 = label1[0, 0].detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=2, ncols=3)
    for ax in axs.flatten():
        ax.axis("off")

    axs[0, 0].imshow(im1[..., im1.shape[2] // 2], cmap="gray")
    axs[0, 1].imshow(im1[:, im1.shape[1] // 2, ...], cmap="gray")
    axs[0, 2].imshow(im1[im1.shape[0] // 2, ...], cmap="gray")
    axs[0, 1].set_title('LF Image', pad=20)

    axs[1, 0].imshow(gt1[..., gt1.shape[2] // 2], cmap="gray")
    axs[1, 1].imshow(gt1[:, gt1.shape[1] // 2, ...], cmap="gray")
    axs[1, 2].imshow(gt1[gt1.shape[0] // 2, ...], cmap="gray")
    axs[1, 1].set_title('HF Image', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(os.getcwd(), "Example Image.png"))
    plt.close()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    device = torch.device("cuda:0")
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_v2=True
    )

    model = model.to(device)

    if vit_model_path is not None:
        model_path = os.path.join(os.getcwd(), vit_model_path)

        print("Loading Weights from the Path {}".format(model_path))
        vit_dict = torch.load(model_path)
        vit_weights = vit_dict["state_dict"]
        model_dict = model.vit.state_dict()

        vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
        model_dict.update(vit_weights)
        model.vit.load_state_dict(model_dict)
        del model_dict, vit_weights, vit_dict
        print("------ViT PR SS weights successfully loaded------")

    if swin_pt_path is not None:
        model_path = os.path.join(os.getcwd(), swin_pt_path)

        print("Loading Weights from the Path {}".format(model_path))
        ssl_dict = torch.load(model_path)
        ssl_weights = ssl_dict["model"]

        # Generate new state dict so it can be loaded to MONAI SwinUNETR Model
        monai_loadable_state_dict = OrderedDict()
        model_prior_dict = model.state_dict()
        model_update_dict = model_prior_dict

        del ssl_weights["encoder.mask_token"]
        del ssl_weights["encoder.norm.weight"]
        del ssl_weights["encoder.norm.bias"]
        del ssl_weights["out.conv.conv.weight"]
        del ssl_weights["out.conv.conv.bias"]

        for key, value in ssl_weights.items():
            if key[:8] == "encoder.":
                if key[8:19] == "patch_embed":
                    new_key = "swinViT." + key[8:]
                else:
                    new_key = "swinViT." + key[8:18] + key[20:]
                monai_loadable_state_dict[new_key] = value
            else:
                monai_loadable_state_dict[key] = value

        model_update_dict.update(monai_loadable_state_dict)
        model.load_state_dict(model_update_dict, strict=True)
        model_final_loaded_dict = model.state_dict()

        # Safeguard test to ensure that weights got loaded successfully
        layer_counter = 0
        for k, _v in model_final_loaded_dict.items():
            if k in model_prior_dict:
                layer_counter = layer_counter + 1

                old_wts = model_prior_dict[k]
                new_wts = model_final_loaded_dict[k]

                old_wts = old_wts.to("cpu").numpy()
                new_wts = new_wts.to("cpu").numpy()
                diff = np.mean(np.abs(old_wts, new_wts))
                print("Layer {}, the update difference is: {}".format(k, diff))
                if diff == 0.0:
                    print("Warning: No difference found for layer {}".format(k))
        print("Total updated layers {} / {}".format(layer_counter, len(model_prior_dict)))
        print("------SwinUNETR Pretrained Weights (SSL) successfully loaded------")

    if swin_model_path is not None:
        model_path = os.path.join(os.getcwd(), swin_model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("------SwinUNETR weights successfully loaded------")

    epoch_loss_values = []
    step_loss_values = []
    epoch_ssim_loss_values = []
    epoch_adv_loss_values = []
    epoch_per_loss_values = []
    epoch_recon_loss_values = []
    val_loss_values = []
    best_val_loss = 1e10

    recon_loss = MSELoss()
    validation_loss = L1Loss()
    structuralsim_loss = SSIMLoss(spatial_dims=3)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    optimizer_d = torch.optim.Adam(params=model.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-06)
    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
    discriminator.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = {
        'scheduler': WarmupCosineSchedule(optimizer, warmup_steps=warmup, t_total=20000)
                    }

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        discriminator.train()
        epoch_loss = 0
        epoch_ssim_loss = 0
        epoch_adv_loss = 0
        epoch_per_loss = 0
        epoch_recon_loss = 0
        val_loss = 0

        step = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=300)
        progress_bar.set_description(f"Train epoch {epoch}")
        for step, batch_data in progress_bar:
            step += 1
            start_time = time.time()

            inputs1, gt1 = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()

            outputs_v1 = model(inputs1)

            r_loss = recon_loss(outputs_v1, gt1)
            p_loss = loss_perceptual(outputs_v1, gt1)
            ssim_loss = structuralsim_loss(outputs_v1, gt1)

            total_loss = p_loss * perceptual_weight + ssim_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(outputs_v1.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                total_loss = p_loss * perceptual_weight + ssim_loss
                total_loss += adv_weight * generator_loss

            if epoch > autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(outputs_v1.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

                logits_real = discriminator(gt1.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss
                loss_d.backward()
                optimizer_d.step()

            total_loss.backward()
            optimizer.step()
            step_loss_values.append(total_loss.item())

            epoch_ssim_loss += ssim_loss.item()
            epoch_recon_loss += r_loss.item()
            epoch_per_loss += p_loss.item()
            epoch_loss += total_loss.item()

            if epoch > autoencoder_warm_up_n_epochs:
                epoch_adv_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "ssim_loss": epoch_ssim_loss / (step + 1),
                    "MSE_loss": epoch_recon_loss / (step + 1),
                    "percep_loss": epoch_per_loss / (step + 1),
                    "adver_loss": epoch_adv_loss / (step + 1),
                    "Total_loss": epoch_loss / (step + 1),
                }
            )

        scheduler = lr_scheduler['scheduler']
        scheduler.step()

        epoch_ssim_loss_values.append(epoch_ssim_loss)
        epoch_recon_loss_values.append(epoch_recon_loss)
        epoch_per_loss_values.append(epoch_per_loss)
        epoch_adv_loss_values.append(epoch_adv_loss)
        epoch_loss_values.append(epoch_loss/ (step + 1))

        plt.clf()
        plt.plot(epoch_loss_values)
        plt.title("Training Loss", fontsize=20)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.savefig(os.path.join(os.getcwd(), "train_loss_plots.png"))
        plt.close(1)

        if epoch % val_interval == 0:
            print("\nEntering VALIDATION for epoch: {}".format(epoch + 1))
            total_val_loss = 0
            val_step = 0
            model.eval()
            for val_batch in val_loader:
                val_step += 1
                start_time = time.time()
                inputs1, gt1 = (
                    val_batch["image"].to(device),
                    val_batch["label"].to(device),
                )

                output_1 = model(inputs1)
                val_loss = validation_loss(output_1, gt1)
                total_val_loss += val_loss.item()
                end_time = time.time()

                plt.clf()
                in1 = inputs1[0, 0].detach().cpu().numpy()
                im1 = output_1[0, 0].detach().cpu().numpy()
                gt_1 = gt1[0, 0].detach().cpu().numpy()

                fig, axs = plt.subplots(nrows=3, ncols=3)
                for ax in axs.flatten():
                    ax.axis("off")
                axs[0, 0].imshow(in1[..., in1.shape[2] // 2], cmap="gray")
                axs[0, 1].imshow(in1[:, in1.shape[1] // 2, ...], cmap="gray")
                axs[0, 2].imshow(in1[in1.shape[0] // 2, ...], cmap="gray")
                axs[0, 1].set_title('LF Image', pad=20)

                axs[1, 0].imshow(im1[..., im1.shape[2] // 2], cmap="gray")
                axs[1, 1].imshow(im1[:, im1.shape[1] // 2, ...], cmap="gray")
                axs[1, 2].imshow(im1[im1.shape[0] // 2, ...], cmap="gray")
                axs[0, 1].set_title('SF Image', pad=20)

                axs[2, 0].imshow(gt_1[..., gt_1.shape[2] // 2], cmap="gray")
                axs[2, 1].imshow(gt_1[:, gt_1.shape[1] // 2, ...], cmap="gray")
                axs[2, 2].imshow(gt_1[gt_1.shape[0] // 2, ...], cmap="gray")
                axs[0, 1].set_title('HF Image', pad=20)

                plt.savefig(os.path.join(os.getcwd(), f"Val_epoch_{epoch}_images.png"))

                total_val_loss /= val_step
                val_loss_values.append(total_val_loss)
                print(f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}")

                if total_val_loss < best_val_loss:
                    print(f"Saving new model based on validation loss {total_val_loss:.4f}")
                    best_val_loss = total_val_loss
                    checkpoint = {"epoch": max_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                    torch.save(checkpoint, os.path.join(os.getcwd(), "best_model.pt"))

            plt.clf()
            plt.plot(val_loss_values)
            plt.title("Validation Loss", fontsize=20)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel("Epochs", fontsize=16)
            plt.ylabel("Loss", fontsize=16)
            plt.savefig(os.path.join(os.getcwd(), "val_loss_plots.png"))
            plt.close(1)

            checkpoint = {"epoch": max_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(os.getcwd(), f"epoch_{epoch}_model.pt"))

    print("Done")