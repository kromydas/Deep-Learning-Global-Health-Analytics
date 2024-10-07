import os
import random
import pandas as pd
import glob as glb
import rasterio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

def get_resnet_18(output_features=1, extraction_layer=None, fine_tune_layers=0, dropout_rate=0.5, features_list=None):

    # Initialize the features_list if not provided
    if features_list is None:
        features_list = []

    def get_module_by_name(model, module_name):
        names = module_name.split('.')
        module = model
        for name in names:
            if hasattr(module, name):
                module = getattr(module, name)
            elif name.isdigit():
                module = module[int(name)]
        return module

    # Instantiate the model with ImageNet weights
    model_res_18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Disable gradient computations for all layers initially (freeze the backbone)
    for name, param in model_res_18.named_parameters():
        param.requires_grad = False

    # Get the layers to fine-tune based on the fine_tune_layers argument
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    if fine_tune_layers > 0:
        layers_to_finetune = layer_names[-fine_tune_layers:]
        for name, param in model_res_18.named_parameters():
            if any(layer in name for layer in layers_to_finetune):
                param.requires_grad = True

    # Ensure the regression head (fc layer) is always trainable, and add Dropout
    model_res_18.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),  # Add Dropout layer here with the specified rate
        nn.Linear(in_features=model_res_18.fc.in_features, out_features=output_features)
    )

    for name, param in model_res_18.fc.named_parameters():
        param.requires_grad = True

    # Hook function to capture output features (use the passed in features_list)
    def hook_conv(module, input, output):
        # print(f"Hook fired, output shape: {output.shape}")
        features_list.append(output.detach())

    # Set the hook on the specified extraction layer
    if extraction_layer:
        print(f"Setting hook for extraction layer: {extraction_layer}")

        # Check if the extraction layer is a high-level layer (e.g., 'layer3' or 'layer4')
        if extraction_layer in layer_names:
            print(f"Extraction layer is a high-level layer: {extraction_layer}")
            # Attach the hook to the last block in the specified layer
            layer = getattr(model_res_18, extraction_layer)
            last_block = layer[-1]  # Get the last block in the Sequential container (e.g., BasicBlock)
            print(f"Attaching hook to the last block in {extraction_layer}")
            last_block.register_forward_hook(lambda module, input, output: hook_conv(module, input, output))
        else:
            # Attach hook to the specific sub-layer
            hook_layer = get_module_by_name(model_res_18, extraction_layer)
            hook_layer.register_forward_hook(lambda module, input, output: hook_conv(module, input, output))
    else:
        print("No extraction layer specified")

    return model_res_18, features_list  # Return the model and features_list


def random_crop(image, crop_size=(160, 160)):
    """
    Perform a random crop on the input image.

    Args:
        image (Tensor): Input image (C x H x W).
        crop_size (tuple): Size of the cropped region (height, width).

    Returns:
        Cropped image.
    """
    _, h, w = image.shape
    ch, cw  = crop_size

    # Ensure crop size is smaller than the image dimensions
    assert h >= ch and w >= cw, "Crop size must be smaller than the image size"

    # Randomly select the top-left corner of the crop
    top  = random.randint(0, h - ch)
    left = random.randint(0, w - cw)

    # Crop the image
    cropped_image = image[:, top:top + ch, left:left + cw]
    return cropped_image

def random_translate(image, max_shift=30):
    _, h, w = image.shape

    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    # Pad the image to accommodate the shift
    pad_left   = max(shift_x, 0)
    pad_right  = max(-shift_x, 0)
    pad_top    = max(shift_y, 0)
    pad_bottom = max(-shift_y, 0)

    # Use 'replicate' mode instead of 'edge'
    padded_image = TF.pad(
        image,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='replicate'
    )

    # Compute the new dimensions after padding
    _, padded_h, padded_w = padded_image.shape

    # Crop the image back to original size, applying the shift
    start_x = pad_left
    end_x   = start_x + w
    start_y = pad_top
    end_y   = start_y + h

    translated_image = padded_image[:, start_y:end_y, start_x:end_x]

    return translated_image


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def extract_features(model, data_loader, features_list, device='cpu'):

    # Reset features_list at the beginning of the function
    features_list.clear()  # Reset the list at the start of each call

    model.to(device)
    model.eval()

    cluster_ids = []
    target_values_list = []  # To store target values

    with torch.no_grad():
        # Unpack the tuple directly from the data_loader
        for images, (cluster_id, aoi, targets) in data_loader:
            images = images.to(device)

            # Extract features by passing images through the model
            _ = model(images)  # Model already has hook attached, features go to features_list

            # Store the cluster IDs and target values
            cluster_ids.extend(cluster_id.tolist())  # Convert to list of cluster IDs
            target_values_list.extend(targets.tolist())  # Convert target values to a list

    # Flatten the feature tensors and concatenate them into one array
    all_features = torch.cat([torch.flatten(f, start_dim=1) for f in features_list], dim=0).cpu().numpy()

    # Return features, cluster IDs, and target values
    return all_features, cluster_ids, target_values_list


def get_image_path(image_path, cid, data_type):
    """
    Constructs and returns the file path for the given cluster ID and data type.
    """
    formatted_cid = str(int(cid))

    # Construct the base path for the specific data type (e.g., 'Rainfall')
    data_type_path = os.path.join(image_path, data_type)

    # Construct the search pattern using glob
    search_pattern = os.path.join(data_type_path, f"*C-{formatted_cid}_{data_type}_*.tif")

    # Use glob to find the file
    matching_files = glb.glob(search_pattern)

    # Ensure only one file is found
    if len(matching_files) == 1:
        return matching_files[0]
    elif len(matching_files) > 1:
        raise ValueError(
            f"Multiple matching files found for Cluster ID {cid} and data type '{data_type}'. Expected only one.")
    else:
        return None  # Return None if no matching file is found


def add_image_paths_for_row(row, image_path):
    """
    Attempts to get the file paths for nightlights, population, and rainfall data for a given row (cluster).
    """
    # Get the index value (Cluster ID)
    cid = row.name

    # Attempt to get file paths for each data type
    nightlights_path = get_image_path(image_path, cid, 'Nightlights')
    population_path  = get_image_path(image_path, cid, 'Population')
    rainfall_path    = get_image_path(image_path, cid, 'Rainfall')

    # Return all paths, whether None or valid
    return pd.Series([nightlights_path, population_path, rainfall_path],
                     index=['nightlights_path', 'population_path', 'rainfall_path'])


def add_geospatial_image_paths(geospatial_df, dataset_config, country_code):
    """
    Adds image paths for each row in the DataFrame based on image paths for each cluster.
    Ensures robustness by flagging any missing files.
    """
    # Construct the base image path
    image_path = os.path.join(dataset_config.AOI_ROOT, f'{country_code}/Image_Tiles/')

    # Apply the function to each row and add the results as new columns
    geospatial_df[['nightlights_path', 'population_path', 'rainfall_path']] = geospatial_df.apply(
        lambda row: add_image_paths_for_row(row, image_path), axis=1
    )

    # Log missing paths, but don't drop the rows immediately
    missing_paths = geospatial_df[['nightlights_path', 'population_path', 'rainfall_path']].isnull().any(axis=1)
    num_missing = missing_paths.sum()

    if num_missing > 0:
        print(f"Warning: {num_missing} rows have missing image paths.")
        print("Missing rows will be retained but can be inspected for further handling.")

    return geospatial_df


def compute_raster_statistics(image_path):

    with rasterio.open(image_path) as src:
        data = src.read(1)  # Assume single channel image
        mean_value = np.mean(data)
        std_value = np.std(data)
    return mean_value, std_value

def add_statistics_to_dataframe(df, data_types):

    for data_type in data_types:
        print(df[f'{data_type.lower()}_path'].head())
        means, stds = [], []
        for _, row in df.iterrows():
            image_path = row[f'{data_type.lower()}_path']
            if image_path and os.path.exists(image_path):  # Ensure path is valid
                mean, std = compute_raster_statistics(image_path)
                means.append(mean)
                stds.append(std)
            else:
                means.append(None)
                stds.append(None)
        df[f'{data_type.lower()}_mean'] = means
        df[f'{data_type.lower()}_std'] = stds
    return df


def assign_cluster_labels_by_matching(geospatial_df, cluster_ids, cluster_labels):
    """
    Assigns cluster labels by matching the cluster IDs from the loader with the geospatial DataFrame.
    This version assumes that cluster_id is unique and is the index of geospatial_df.

    Arguments:
    - geospatial_df: pandas DataFrame with cluster_id as the index.
    - cluster_ids: list of cluster IDs from the data loader (list of integers).
    - cluster_labels: list of cluster labels assigned by the clustering algorithm (same length as cluster_ids).

    Returns:
    - geospatial_df: the DataFrame with an added 'cluster_label' column.
    """

    # Check if 'cluster_label' column already exists in the DataFrame
    if 'cluster_label' in geospatial_df.columns:
        print("'cluster_label' column already exists. Skipping assignment.")
        return geospatial_df  # Return the DataFrame unchanged

    # Ensure the lengths of cluster_ids and cluster_labels are the same
    if len(cluster_ids) != len(cluster_labels):
        raise ValueError("The lengths of cluster_ids and cluster_labels must be the same.")

    # Create a DataFrame to associate cluster_ids with their corresponding cluster_labels
    cluster_mapping_df = pd.DataFrame({
        'cluster_id': cluster_ids,
        'cluster_label': cluster_labels
    })

    # Set the index of cluster_mapping_df to 'cluster_id' for merging
    cluster_mapping_df.set_index('cluster_id', inplace=True)

    # Merge the cluster labels into the geospatial DataFrame based on 'cluster_id'
    geospatial_df = geospatial_df.join(cluster_mapping_df[['cluster_label']], how='left')

    # Check if cluster_label was added
    print("geospatial_df after merging:\n", geospatial_df[['cluster_label']].head())

    # Check for any missing cluster labels
    missing_labels = geospatial_df['cluster_label'].isnull().sum()
    if missing_labels > 0:
        print(f"Warning: {missing_labels} rows in the geospatial DataFrame did not find a matching cluster label.")

    return geospatial_df

def compute_cluster_aggregated_statistics_and_correlations(geospatial_df_copy, cluster_label):
    """
    Computes and normalizes the aggregated statistics (mean and std) for each geospatial data type
    for a given cluster, using the 'cluster_label' column to filter the cluster.
    Also returns the correlations between geospatial data types and survey metrics for the cluster,
    and the vaccination rates for both cluster and non-cluster points.

    Arguments:
    - geospatial_df_copy: pandas DataFrame, containing geospatial data with 'cluster_label' as a column.
    - cluster_label: int or str, the cluster label to filter by.

    Returns:
    - cluster_stats: dict, aggregated and normalized mean and std statistics for the given cluster
    - non_cluster_stats: dict, aggregated and normalized mean and std statistics for points outside the cluster
    - cluster_count: int, number of points in the specified cluster
    - non_cluster_count: int, number of points outside the specified cluster
    - unnormalized_cluster_means: dict, unnormalized means for display
    - unnormalized_non_cluster_means: dict, unnormalized means for display
    - survey_means_cluster: list, mean values for the cluster for survey metrics
    - correlations: dict, correlations between geospatial data types and survey metrics
    - cluster_vaccination_rates: list, raw vaccination rates for points in the cluster
    - non_cluster_vaccination_rates: list, raw vaccination rates for points outside the cluster
    """

    # Filter out points that belong to the given cluster using the 'cluster_label' column
    cluster_data = geospatial_df_copy[geospatial_df_copy['cluster_label'] == cluster_label].copy()

    # Data outside the cluster (non-cluster)
    non_cluster_data = geospatial_df_copy[geospatial_df_copy['cluster_label'] != cluster_label].copy()

    # Explicitly define the columns that contain geospatial data
    geospatial_mean_cols = ['nightlights_mean', 'population_mean', 'rainfall_mean']

    # Survey metrics to compute correlations with
    survey_metrics = ['fraction_dpt3_vaccinated',
                      'fraction_with_electricity',
                      'fraction_with_fresh_water',
                      'fraction_with_radio',
                      'fraction_with_tv',
                      'mean_wealth_index']

    # Compute correlations between geospatial data and survey metrics for the cluster
    correlations = {}
    for geo_col in geospatial_mean_cols:
        correlations[geo_col] = {}
        for survey_col in survey_metrics:
            # Compute Pearson correlation
            correlation = cluster_data[geo_col].corr(cluster_data[survey_col])
            correlations[geo_col][survey_col] = correlation

    # Compute min and max for each geospatial data type to normalize
    min_max_values = {
        col: (geospatial_df_copy[col].min(), geospatial_df_copy[col].max())
        for col in geospatial_mean_cols
    }

    # Function to normalize data to [0, 1] range
    def normalize(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val) if max_val != min_val else data

    # Compute aggregated statistics for the cluster points
    cluster_stats = {}
    non_cluster_stats = {}
    unnormalized_cluster_means = {}
    unnormalized_non_cluster_means = {}

    for col in geospatial_mean_cols:
        min_val, max_val = min_max_values[col]

        # Unnormalized means for display
        unnormalized_cluster_means[col] = cluster_data[col].mean()
        unnormalized_non_cluster_means[col] = non_cluster_data[col].mean()

        # Normalized means and standard deviations
        cluster_stats[col] = {
            'mean': normalize(cluster_data[col].mean(), min_val, max_val),
            'std': normalize(cluster_data[col].std(), min_val, max_val)
        }
        non_cluster_stats[col] = {
            'mean': normalize(non_cluster_data[col].mean(), min_val, max_val),
            'std': normalize(non_cluster_data[col].std(), min_val, max_val)
        }

    # Get the number of points in the cluster and non-cluster
    cluster_count = cluster_data.shape[0]
    non_cluster_count = non_cluster_data.shape[0]

    # Compute the means for the specified survey metrics for the cluster
    survey_means_cluster = [cluster_data['fraction_dpt3_vaccinated'].mean(),
                            cluster_data['fraction_with_electricity'].mean(),
                            cluster_data['fraction_with_fresh_water'].mean(),
                            cluster_data['fraction_with_radio'].mean(),
                            cluster_data['fraction_with_tv'].mean(),
                            cluster_data['mean_wealth_index'].mean()]

    # Extract the raw vaccination rates for both cluster and non-cluster
    cluster_vaccination_rates = cluster_data['fraction_dpt3_vaccinated'].tolist()
    non_cluster_vaccination_rates = non_cluster_data['fraction_dpt3_vaccinated'].tolist()

    return (cluster_stats, non_cluster_stats, cluster_count, non_cluster_count,
            unnormalized_cluster_means, unnormalized_non_cluster_means,
            survey_means_cluster, correlations, cluster_vaccination_rates, non_cluster_vaccination_rates)
