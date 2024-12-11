import os
import json
import re
from collections import Counter
import warnings

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np
import rasterio

def custom_transforms(image, mean, std, img_size):
    """
    Custom transformation function for preprocessing images.

    Args:
        image (torch.Tensor or np.ndarray): Input image.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
        img_size (tuple): Target size for resizing.

    Returns:
        torch.Tensor: Transformed image.
    """
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)

    # Check and adjust shape to [C, H, W]
    if image.ndim == 3 and image.shape[-1] == 6:  # if [H, W, C]
        image = image.permute(2, 0, 1)  # Convert to [C, H, W]

    # Resize image
    resize_transform = T.Resize(img_size, antialias=True)
    image = resize_transform(image)

    # Normalize image
    normalize_transform = T.Normalize(mean=mean, std=std)
    image = normalize_transform(image)

    return image

class HLSFlexibleBandSatDataset(Dataset):

    def __init__(self, root_dir, partition_map_path, num_channels=6, selected_targets=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            partition_map_path (string): Path to the partition map JSON file.
            num_channels (int): Number of channels to process (either 3 or 6).
            selected_targets (list): List of target names to retrieve.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if num_channels not in [3, 6]:
            raise ValueError("num_channels must be either 3 (RGB) or 6 (RGB + IR).")

        self.root_dir = root_dir
        self.partition_map_path = partition_map_path
        self.num_channels = num_channels
        self.transform = transform
        self.selected_targets = selected_targets or ['mean_wealth_index']
        self.target_values = {}
        self.aoi_counter = Counter()  # Counter to track the number of clusters per AOI

        # Load the partition mapping from the JSON file
        with open(partition_map_path, 'r') as f:
            self.partition_map = json.load(f)

        # Load target values for each country in the partition map
        for country_code in self.partition_map.keys():
            target_json_path = os.path.join(self.root_dir, country_code, "Targets", "targets.json")
            with open(target_json_path, 'r') as f:
                self.target_values[country_code] = json.load(f)

        # Build the file list and cluster IDs
        self.file_list, self.cluster_ids = self._build_file_list()

    def _get_target_values(self, cluster_id, country_code):
        """
        Helper function to retrieve the target values for a given cluster ID and country code.
        Ensures no missing values exist.
        """
        cluster_id_str = str(cluster_id)
        target_data = self.target_values.get(country_code, {}).get("clusters", {})
        cluster_data = target_data.get(cluster_id_str)

        if cluster_data is None:
            raise ValueError(f"Error: Cluster ID {cluster_id_str} not found in targets for AOI {country_code}")

        target_values = []
        for target in self.selected_targets:
            target_value = cluster_data.get(target, None)
            if target_value is None:
                raise ValueError(
                    f"Error: Target '{target}' for cluster_id {cluster_id_str} in AOI {country_code} is missing or set to None.")
            target_values.append(target_value)

        return target_values

    def _build_file_list(self):
        cluster_files = {}  # Key: (country_code, cluster_id), Value: file_path

        for country_code, cluster_ids in self.partition_map.items():
            image_tiles_path = os.path.join(self.root_dir, country_code, "Satellite_Tiles")

            print(f"Processing AOI: {country_code}, {len(cluster_ids)} clusters")

            for subdir, dirs, files in os.walk(image_tiles_path):
                for file in files:
                    match = re.search(r"C-(\d+)", file)
                    if match:
                        cluster_id = int(match.group(1))
                        if cluster_id in cluster_ids:
                            key = (country_code, cluster_id)
                            cluster_files.setdefault(key, []).append(os.path.join(subdir, file))

        file_list = []
        cluster_ids = []
        skipped_clusters = 0

        for key, file_paths in cluster_files.items():
            if not file_paths:
                print(f"Warning: No files found for cluster {key}")
                skipped_clusters += 1
                continue

            file_list.append(tuple(file_paths))
            cluster_ids.append(key)

            country_code = key[0]
            self.aoi_counter[country_code] += 1

        return file_list, cluster_ids

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_file = self.file_list[idx]
        if isinstance(sample_file, tuple):  # If it’s a tuple, take the first item
            sample_file = sample_file[0]

        country_code, cluster_id = self.cluster_ids[idx]
        aoi = country_code

        with rasterio.open(sample_file) as src:

            # Read only the specified number of channels (first `num_channels`)
            data = src.read(indexes=range(1, self.num_channels + 1), out_dtype="float32")

            # Ensure the data has the correct number of channels
            if data.shape[0] != self.num_channels:
                raise ValueError(f"Expected {self.num_channels} channels but got {data.shape[0]} in file {sample_file}")

            # Check for potential data issues before normalization
            if np.isnan(data).any():
                print(f"NaN values detected in file {sample_file} for cluster {cluster_id}")
                raise ValueError(f"NaN values found in file {sample_file}")

            for i in range(data.shape[0]):
                min_val, max_val = data[i].min(), data[i].max()
                if max_val - min_val < 1e-6:
                    warnings.warn(f"Channel {i} in file {sample_file} has nearly constant values.")

                if min_val == max_val:
                    warnings.warn(f"Channel {i} in file {sample_file} has identical min and max values ({min_val}).")
                    data[i] = np.zeros_like(data[i])

        image = torch.from_numpy(data).float()

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        # Retrieve target values
        target_values = self._get_target_values(cluster_id, country_code)

        return image, (cluster_id, aoi, torch.tensor(target_values, dtype=torch.float32))

# class HLSFlexibleBandSatDataset(Dataset):
#
#     def __init__(self, root_dir, partition_map_path, num_channels=6, selected_targets=None, transform=None, **transform_args):
#         """
#         Args:
#             root_dir (string): Directory with all the images.
#             partition_map_path (string): Path to the partition map JSON file.
#             num_channels (int): Number of channels to process (either 3 or 6).
#             selected_targets (list): List of target names to retrieve.
#             transform (callable, optional): Optional transform to be applied on a sample.
#             **transform_args: Additional arguments to pass to the transform function.
#         """
#         if num_channels not in [3, 6]:
#             raise ValueError("num_channels must be either 3 (RGB) or 6 (RGB + IR).")
#
#         self.root_dir = root_dir
#         self.partition_map_path = partition_map_path
#         self.num_channels = num_channels
#         self.transform = transform
#         self.transform_args = transform_args  # Store additional arguments for the transform
#         self.selected_targets = selected_targets or ['mean_wealth_index']
#         self.target_values = {}
#         self.aoi_counter = Counter()  # Counter to track the number of clusters per AOI
#
#         # Load the partition mapping from the JSON file
#         with open(partition_map_path, 'r') as f:
#             self.partition_map = json.load(f)
#
#         # Load target values for each country in the partition map
#         for country_code in self.partition_map.keys():
#             target_json_path = os.path.join(self.root_dir, country_code, "Targets", "targets.json")
#             with open(target_json_path, 'r') as f:
#                 self.target_values[country_code] = json.load(f)
#
#         # Build the file list and cluster IDs
#         self.file_list, self.cluster_ids = self._build_file_list()
#
#     def _get_target_values(self, cluster_id, country_code):
#         """
#         Helper function to retrieve the target values for a given cluster ID and country code.
#         Ensures no missing values exist.
#         """
#         cluster_id_str = str(cluster_id)
#         target_data = self.target_values.get(country_code, {}).get("clusters", {})
#         cluster_data = target_data.get(cluster_id_str)
#
#         if cluster_data is None:
#             raise ValueError(f"Error: Cluster ID {cluster_id_str} not found in targets for AOI {country_code}")
#
#         target_values = []
#         for target in self.selected_targets:
#             target_value = cluster_data.get(target, None)
#             if target_value is None:
#                 raise ValueError(
#                     f"Error: Target '{target}' for cluster_id {cluster_id_str} in AOI {country_code} is missing or set to None.")
#             target_values.append(target_value)
#
#         return target_values
#
#     def _build_file_list(self):
#         cluster_files = {}  # Key: (country_code, cluster_id), Value: file_path
#
#         for country_code, cluster_ids in self.partition_map.items():
#             image_tiles_path = os.path.join(self.root_dir, country_code, "Satellite_Tiles")
#
#             print(f"Processing AOI: {country_code}, {len(cluster_ids)} clusters")
#
#             for subdir, dirs, files in os.walk(image_tiles_path):
#                 for file in files:
#                     match = re.search(r"C-(\d+)", file)
#                     if match:
#                         cluster_id = int(match.group(1))
#                         if cluster_id in cluster_ids:
#                             key = (country_code, cluster_id)
#                             cluster_files.setdefault(key, []).append(os.path.join(subdir, file))
#
#         file_list = []
#         cluster_ids = []
#         skipped_clusters = 0
#
#         for key, file_paths in cluster_files.items():
#             if not file_paths:
#                 print(f"Warning: No files found for cluster {key}")
#                 skipped_clusters += 1
#                 continue
#
#             file_list.append(tuple(file_paths))
#             cluster_ids.append(key)
#
#             country_code = key[0]
#             self.aoi_counter[country_code] += 1
#
#         return file_list, cluster_ids
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         sample_file = self.file_list[idx]
#         if isinstance(sample_file, tuple):  # If it’s a tuple, take the first item
#             sample_file = sample_file[0]
#
#         country_code, cluster_id = self.cluster_ids[idx]
#         aoi = country_code
#
#         with rasterio.open(sample_file) as src:
#             # Read only the specified number of channels (first `num_channels`)
#             data = src.read(indexes=range(1, self.num_channels + 1), out_dtype="float32")
#
#             # Ensure the data has the correct number of channels
#             if data.shape[0] != self.num_channels:
#                 raise ValueError(f"Expected {self.num_channels} channels but got {data.shape[0]} in file {sample_file}")
#
#             # Check for potential data issues before normalization
#             if np.isnan(data).any():
#                 print(f"NaN values detected in file {sample_file} for cluster {cluster_id}")
#                 raise ValueError(f"NaN values found in file {sample_file}")
#
#             for i in range(data.shape[0]):
#                 min_val, max_val = data[i].min(), data[i].max()
#                 if max_val - min_val < 1e-6:
#                     warnings.warn(f"Channel {i} in file {sample_file} has nearly constant values.")
#
#                 if min_val == max_val:
#                     warnings.warn(f"Channel {i} in file {sample_file} has identical min and max values ({min_val}).")
#                     data[i] = np.zeros_like(data[i])
#
#         image = torch.from_numpy(data).float()
#
#         # Apply transforms if specified
#         if self.transform:
#             image = self.transform(image, **self.transform_args)  # Pass additional arguments
#
#         # Retrieve target values
#         target_values = self._get_target_values(cluster_id, country_code)
#
#         return image, (cluster_id, aoi, torch.tensor(target_values, dtype=torch.float32))
