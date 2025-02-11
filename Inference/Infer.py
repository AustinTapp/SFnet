import os
import json
import time
import shutil
import tempfile
import sys
import pdb
import gc

import glob
import matplotlib.pyplot as plt
import torch
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from collections import OrderedDict
from monai import transforms
from monai.config import print_config
from monai.data import DataLoader, Dataset, list_data_collate, decollate_batch
from monai.utils import first, set_determinism
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from functools import partial

obj = None
gc.collect()
torch.cuda.empty_cache()

def mask(image):
    # Thresholding to create a binary mask
    threshold_value = 0.2
    binary_mask = image > threshold_value

    # Connected component labeling to separate regions
    connected_components = sitk.ConnectedComponent(binary_mask)

    # Filter out small regions
    minimum_region_size = 1000  # Adjust as needed
    filtered_components = sitk.RelabelComponent(connected_components, minimum_region_size)
    closed_mask = sitk.BinaryMorphologicalClosing(filtered_components, 5)

    segmentation = sitk.Cast(closed_mask, sitk.sitkFloat32)
    return segmentation


if __name__ == "__main__":
    root_dir = str(os.getcwd())
    LF_data = sorted(glob.glob(root_dir + "\\LF\\Base\\*.nii.gz"))
    HF_data = sorted(glob.glob(root_dir + "\\HF\\Base\\*.nii.gz"))

    LF_output_path = os.path.join(root_dir + "\\LF\\PP")
    HF_output_path = os.path.join(root_dir + "\\HF\\PP") # for HF GT comparison

    model_path = os.path.join(root_dir + "\\Models\\SFNet_Final_Weights.pt")
    SF_output_path = os.path.join(root_dir + "\\SF")

    if not os.path.exists(SF_output_path):
        os.makedirs(SF_output_path)

    LF_list = []
    HF_list = []
    for idx, path in enumerate(LF_data):
        image_path = LF_data
        data_point = {"image": image_path[idx]}
        LF_list.append(data_point)

    for idx, path in enumerate(HF_data):
        image_path = HF_data
        data_point = {"image": image_path[idx]}
        HF_list.append(data_point)

    LF_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(192, 192, 192)),
        transforms.SpatialPadd(keys=["image"], spatial_size=(192, 192, 192), mode='empty'),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1),
    ]
                                        )

    HF_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(192, 192, 192)),
        transforms.SpatialPadd(keys=["image"], spatial_size=(192, 192, 192), mode='empty'),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1),
    ]
                                        )

    device = torch.device(0)

    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_v2=True
    )

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    model.to(device)
    model.eval()

    model_inferer = partial(
        sliding_window_inference,
        progress=True,
        roi_size=[96, 96, 96],
        sw_batch_size=12,
        predictor=model,
        overlap=0.9,
        mode='gaussian',
        sigma_scale=0.2
    )

    #To achieve perfect image overlap for comparisons, LF (and/or HF) images should be processed.
    #The LF and HF processings are below.

    for i in range(len(LF_list)):
        LF_image = LF_list[i]
        pred_img_name = LF_image["image"].split("\\")[-1].split(".")[0]
        batch_data = LF_transforms(LF_image)
        batch_data = list_data_collate([batch_data])
        low_field = batch_data['image'].to(device)

        print(f"Processing LF image {pred_img_name}")
        LF = low_field

        LF.applied_operations = batch_data['image'].applied_operations
        batch_data.update({'image': LF})
        batch_data = [LF_transforms.inverse(i) for i in decollate_batch(batch_data)]

        LF_prediction = batch_data[0]["image"]
        LF_pred = LF_prediction[0].detach().cpu().numpy()
        LF_pred = LF_pred.astype(np.float32)
        LF_pred = np.swapaxes(LF_pred, 0, 2)
        LF_pred = sitk.GetImageFromArray(LF_pred)

        original = sitk.ReadImage(LF_image['image'])
        LF_pred.SetOrigin(original.GetOrigin())
        LF_pred.SetDirection(original.GetDirection())

        sitk.WriteImage(LF_pred, os.path.join(LF_output_path, f"{pred_img_name}_OG.nii.gz"))

    if HF_list:
        for i in range(len(HF_list)):
            HF_image = HF_list[i]
            pred_img_name = HF_image["image"].split("\\")[-1].split(".")[0]
            batch_data = HF_transforms(HF_image)
            batch_data = list_data_collate([batch_data])
            high_field = batch_data['image'].to(device)

            print(f"Processing HF image {pred_img_name}")
            HF = high_field

            HF.applied_operations = batch_data['image'].applied_operations
            batch_data.update({'image': HF})
            batch_data = [HF_transforms.inverse(i) for i in decollate_batch(batch_data)]

            HF_prediction = batch_data[0]["image"]
            HF_pred = HF_prediction[0].detach().cpu().numpy()
            HF_pred = HF_pred.astype(np.float32)
            HF_pred = np.swapaxes(HF_pred, 0, 2)
            HF_pred = sitk.GetImageFromArray(HF_pred)

            original = sitk.ReadImage(HF_image['image'])
            HF_pred.SetOrigin(original.GetOrigin())
            HF_pred.SetDirection(original.GetDirection())

            sitk.WriteImage(HF_pred, os.path.join(HF_output_path, f"{pred_img_name}_GT.nii.gz"))

    with torch.no_grad():
        for i in range(len(LF_list)):
            LF_image = LF_list[i]
            pred_img_name = LF_image["image"].split("\\")[-1].split(".")[0]
            batch_data = LF_transforms(LF_image)
            batch_data = list_data_collate([batch_data])
            low_field = batch_data['image'].to(device)

            print(f"Superfielding image {pred_img_name}")
            sHF = model_inferer(low_field)

            sHF.applied_operations = batch_data['image'].applied_operations
            batch_data.update({'image': sHF})
            batch_data = [LF_transforms.inverse(i) for i in decollate_batch(batch_data)]

            sHF_prediction = batch_data[0]["image"]
            sHF_pred = sHF_prediction[0].detach().cpu().numpy()
            sHF_pred = sHF_pred.astype(np.float32)
            sHF_pred = np.swapaxes(sHF_pred, 0, 2)

            LF_og = sitk.ReadImage(os.path.join(LF_output_path, pred_img_name + "_OG.nii.gz"))
            
            sHF_pred = sitk.GetImageFromArray(sHF_pred)

            sHF_pred.SetOrigin(LF_og.GetOrigin())
            sHF_pred.SetDirection(LF_og.GetDirection())

            LF_mask = mask(LF_og)

            sHF_pred_array = sitk.GetArrayFromImage(sHF_pred)
            LF_mask_array = sitk.GetArrayFromImage(LF_mask)
            sHF_pred_array *= LF_mask_array

            sHF_pred = sitk.GetImageFromArray(sHF_pred_array)
            sHF_pred.SetOrigin(LF_og.GetOrigin())
            sHF_pred.SetDirection(LF_og.GetDirection())

            sitk.WriteImage(sHF_pred, os.path.join(SF_output_path, f"{pred_img_name}_SF.nii.gz"))

            print(f"\nInferring image {pred_img_name} success!")
            i=+1

    print("Done")