import os
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, Orientationd, CropForegroundd, 
    SpatialPadd, EnsureTyped, CenterSpatialCropd, Transform
)
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import exposure

class AddChannelTransform:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = torch.unsqueeze(torch.tensor(data[key]), 0)  # Adds a channel dimension
        return data

class EqualizeAndRescaleHistTransform(Transform):
    """
    Apply histogram equalization to each slice of a 3D volume tensor and rescale it to the original range.
    """

    def __init__(self, keys, save_path='images after HistoEqual'):
        self.keys = keys
        self.save_path = save_path
        #os.makedirs(self.save_path, exist_ok=True)

    def save_image(self, data, key, status=""):
      """
      Save the image data.

      Args:
          data (ndarray): The data to save.
          key (str): The key associated with the data.
      """
      c, h, w, d = data.shape
      middle_slice = d // 2
      for i in range(c):  # Loop over channels
          img = data[i, :, :, middle_slice]  # Save the middle slice
          plt.imshow(img, cmap='gray')
          # plt.title(f'{key}_after_channel_{i}')
          plt.axis('off')
          #plt.savefig(os.path.join(self.save_path, f'{key}_{status}_histoEqual.png'))
          plt.close()

    def __call__(self, data):
        for key in self.keys:
            volume_tensor = data[key]
            volume_np = volume_tensor.cpu().numpy()
            equalized_rescaled_volume_np = np.zeros_like(volume_np)

            # Save image after transform
            # self.save_image(volume_tensor, key, "before")

            for i in range(volume_np.shape[0]):
                slice_np = volume_np[i, :, :]
                original_min, original_max = slice_np.min(), slice_np.max()
                equalized_slice_np = exposure.equalize_hist(slice_np)
                rescaled_slice_np = equalized_slice_np * (original_max - original_min) + original_min
                equalized_rescaled_volume_np[i, :, :] = rescaled_slice_np


            # Save image after transform
            # self.save_image(equalized_rescaled_volume_tensor, key, "after")
            equalized_rescaled_volume_tensor = torch.from_numpy(equalized_rescaled_volume_np).to(volume_tensor.device)
            data[key] = equalized_rescaled_volume_tensor

        return data


class UnsharpMasking3D(Transform):
    """
    Apply unsharp masking for 3D medical images to enhance edge contrast.
    Processes each 2D slice along the specified axis independently.
    """
    def __init__(self, keys, sigma=1.0, strength=1.5, axis=0, save_path='images after unsharp'):
        """
        Args:
            keys (list of str): Keys of the corresponding items to be transformed.
            sigma (float): Standard deviation for Gaussian kernel used in blurring.
            strength (float): Scaling factor for adding back the unsharp mask.
            axis (int): Axis of slicing, typically 0 for medical volumes.
        """
        super().__init__()
        self.keys = keys
        self.sigma = sigma
        self.strength = strength
        self.axis = axis
        self.save_path = save_path
        #os.makedirs(self.save_path, exist_ok=True)

    def save_image(self, data, key):
      """
      Save the image data.

      Args:
          data (ndarray): The data to save.
          key (str): The key associated with the data.
      """
      c, h, w, d = data.shape
      middle_slice = d // 2
      for i in range(c):  # Loop over channels
          img = data[i, :, :, middle_slice]  # Save the middle slice
          plt.imshow(img, cmap='gray')
          # plt.title(f'{key}_after_channel_{i}')
          plt.axis('off')
          #plt.savefig(os.path.join(self.save_path, f'{key}_after_unsharp.png'))
          plt.close()

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                volume = data[key]

                # Process each slice along the specified axis
                output = np.zeros_like(volume)
                for i in range(volume.shape[self.axis]):
                    if self.axis == 0:
                        slice_data = volume[i, :, :]
                    elif self.axis == 1:
                        slice_data = volume[:, i, :]
                    else:
                        slice_data = volume[:, :, i]

                    # Apply Gaussian blur to the slice
                    blurred = gaussian_filter(slice_data, sigma=self.sigma)

                    # Calculate the unsharp mask
                    mask = slice_data - blurred

                    # Enhance the slice by adding the scaled mask
                    enhanced_slice = slice_data + self.strength * mask

                    # Place the enhanced slice back into the volume
                    if self.axis == 0:
                        output[i, :, :] = enhanced_slice
                    elif self.axis == 1:
                        output[:, i, :] = enhanced_slice
                    else:
                        output[:, :, i] = enhanced_slice

                 # Save image after transform
                # self.save_image(output, key)

                # Update the data dictionary
                data[key] = output

        return data



class NormalizeAndRescaleTransformFirst(Transform):
    """
    Apply normalization to each slice of a 3D volume tensor and rescale it to a specified range.
    """

    def __init__(self, keys, min_val=0.0, max_val=1.0):
        """
        Args:
            keys (list of str): Keys of the corresponding items to be transformed.
            min_val (float): Minimum value of the desired output range.
            max_val (float): Maximum value of the desired output range.
        """
        self.keys = keys
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data):
        for key in self.keys:
            volume_tensor = data[key]
            volume_np = volume_tensor.cpu().numpy()
            normalized_rescaled_volume_np = np.zeros_like(volume_np)

            for i in range(volume_np.shape[0]):
                slice_np = volume_np[i, :, :]
                original_min, original_max = slice_np.min(), slice_np.max()
                normalized_slice_np = (slice_np - original_min) / (original_max - original_min)
                rescaled_slice_np = normalized_slice_np * (self.max_val - self.min_val) + self.min_val
                normalized_rescaled_volume_np[i, :, :] = rescaled_slice_np

            normalized_rescaled_volume_tensor = torch.from_numpy(normalized_rescaled_volume_np).to(volume_tensor.device)
            data[key] = normalized_rescaled_volume_tensor

        return data
    


    
def get_data():

    # Prepare the dataset
    data_dir = 'data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    patients = [patient for patient in os.listdir(data_dir) if patient.startswith("BraTS20_") and len(os.listdir(os.path.join(data_dir, patient))) == 5]

    data_dicts = [{
        "filename": patient,
        "t1": os.path.join(data_dir, patient, patient+"_t1.nii"),
        "t2": os.path.join(data_dir, patient, patient+"_t2.nii"),
        "t1ce": os.path.join(data_dir, patient, patient+"_t1ce.nii"),
        "flair": os.path.join(data_dir, patient, patient+"_flair.nii"),
        "seg": os.path.join(data_dir, patient, patient+"_seg.nii"),
    } for patient in patients if os.path.isdir(os.path.join(data_dir, patient))]

    # Sorting the dictionaries by 'filename'
    data_dicts = sorted(data_dicts, key=lambda x: x['filename'])
    
    #shuffle the data
    np.random.shuffle(data_dicts)
    
    test_size = int(0.2 * len(data_dicts))

    # Split the data_dicts for training and validation
    train_files, val_files = data_dicts[:-test_size], data_dicts[-test_size:]

    return train_files, val_files

def get_transforms():
    
    # Define the keys
    keys = ["t1", "t2", "t1ce", "flair", "seg"]
        
    # Instantiate your transformation with the KMeansTransform included
    transforms = Compose([
        LoadImaged(keys=keys),
        AddChannelTransform(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        CropForegroundd(keys=keys, source_key="t1"),
        SpatialPadd(keys=keys, spatial_size=(160, 160, 80), mode="constant"),
        CenterSpatialCropd(keys=keys, roi_size=(160, 160, 80)),
        EnsureTyped(keys=keys),
        NormalizeAndRescaleTransformFirst(keys=["t1", "t2", "t1ce", "flair"]),
        EqualizeAndRescaleHistTransform(keys=["t1", "t2", "t1ce", "flair"]), 
        UnsharpMasking3D(keys=["t1", "t2", "t1ce", "flair"], sigma=1.0, strength=1.5, axis=0),
        ])
    
    return transforms
