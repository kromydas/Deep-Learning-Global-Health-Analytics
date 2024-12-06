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
    """
        Instantiates a ResNet-18 model for regression tasks, optionally fine-tuning specific layers and extracting features.

        Parameters:
        -----------
        output_features : int, optional (default=1)
            The number of output features for the final fully connected layer, typically used for regression tasks.
        extraction_layer : str, optional
            The layer from which features will be extracted. It can be a high-level layer (e.g., 'layer3', 'layer4') or a specific sub-layer.
        fine_tune_layers : int, optional (default=0)
            Number of high-level layers to fine-tune starting from the deepest. For example, setting `fine_tune_layers=2` will fine-tune 'layer4' and 'layer3'.
        dropout_rate : float, optional (default=0.5)
            Dropout rate for the fully connected layer to prevent overfitting.
        features_list : list, optional
            A list to store the extracted features. If None, an empty list will be initialized.

        Returns:
        --------
        model_res_18 : torch.nn.Module
            The ResNet-18 model configured for regression, with optional layers set to be fine-tuned.
        features_list : list
            A list containing feature maps extracted from the specified extraction layer during forward passes.

        Notes:
        ------
        - The model is loaded with ImageNet pre-trained weights (`ResNet18_Weights.IMAGENET1K_V1`).
        - By default, all layers are frozen except for those specified for fine-tuning and the final fully connected (fc) layer.
        - If an extraction layer is specified, a hook is set to capture the output of that layer during forward propagation.
        - Dropout is added before the final fully connected layer to reduce overfitting risk.

        Example:
        --------
        model, features = get_resnet_18(output_features=1, extraction_layer='layer4.1.conv2', fine_tune_layers=2, dropout_rate=0.3)
        """
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
    Perform a random crop on the input image tensor.

    Parameters:
    -----------
    image : Tensor
        The input image tensor with dimensions (C x H x W), where C is the number of channels,
        H is the height, and W is the width.
    crop_size : tuple, optional (default=(160, 160))
        The size of the cropped region in the format (height, width).

    Returns:
    --------
    cropped_image : Tensor
        The randomly cropped image tensor with dimensions (C x crop_height x crop_width).

    Notes:
    ------
    - The function ensures that the crop size is smaller than or equal to the dimensions of the input image.
    - The top-left corner of the crop is selected randomly to introduce variability.

    Example:
    --------
    cropped_img = random_crop(image, crop_size=(100, 100))
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
    """
    Apply a random translation to the input image tensor.

    Parameters:
    -----------
    image : Tensor
        The input image tensor with dimensions (C x H x W), where C is the number of channels,
        H is the height, and W is the width.
    max_shift : int, optional (default=30)
        The maximum number of pixels to shift the image in both x and y directions.

    Returns:
    --------
    translated_image : Tensor
        The translated image tensor, maintaining the original dimensions (C x H x W).

    Notes:
    ------
    - The function pads the image to accommodate the shift and then crops it back to the original dimensions.
    - Padding is performed using 'replicate' mode, which replicates the border values to fill the padding region.
    - The shift values are randomly selected within the range [-max_shift, max_shift] for both x and y directions.

    Example:
    --------
    translated_img = random_translate(image, max_shift=20)
    """
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
    """
    Implements early stopping to terminate training when the validation loss stops improving.

    Parameters:
    -----------
    patience : int, optional (default=10)
        The number of epochs to wait after the last time validation loss improved before stopping the training.
    min_delta : float, optional (default=0.001)
        The minimum change in the monitored quantity to qualify as an improvement.

    Attributes:
    -----------
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    counter : int
        Number of epochs since the last improvement in validation loss.
    best_loss : float or None
        The best observed validation loss so far.
    early_stop : bool
        Whether early stopping should occur.

    Example:
    --------
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    for epoch in range(epochs):
        # Training code...
        val_loss = evaluate_validation_loss()
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    """
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
    """
    Extract features from a model using a data loader and a pre-defined hook function.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model from which features are to be extracted. The model should already have a hook attached to capture the features.
    data_loader : DataLoader
        A PyTorch DataLoader that provides batches of data to the model for feature extraction.
    features_list : list
        A list to which the extracted features are appended. This list will be reset at the beginning of each function call.
    device : str, optional (default='cpu')
        The device on which the model and data should be processed, either 'cpu' or 'cuda'.

    Returns:
    --------
    all_features : numpy.ndarray
        A NumPy array containing the concatenated features extracted from all batches. Each row corresponds to a feature vector for an input image.
    cluster_ids : list
        A list of cluster IDs corresponding to each input image in the dataset.
    target_values_list : list
        A list of target values corresponding to each input image in the dataset.

    Notes:
    ------
    - The function assumes that the model has a hook function attached, which appends the extracted features to `features_list` during forward passes.
    - `features_list` is cleared at the beginning of the function to avoid accumulating features across multiple calls.
    - No gradients are calculated during the feature extraction, ensuring efficient inference.

    Example:
    --------
    model, features_list = get_resnet_18(output_features=1, extraction_layer='layer4', fine_tune_layers=2)
    features, cluster_ids, targets = extract_features(model, data_loader, features_list, device='cuda')
    """
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

    Parameters:
    -----------
    image_path : str
        The base directory where the image files are stored.
    cid : int or float
        The cluster ID for which the image path is being constructed. The ID is formatted as an integer.
    data_type : str
        The type of data to be retrieved (e.g., 'Rainfall', 'Population', etc.).

    Returns:
    --------
    str or None
        The file path to the image corresponding to the given cluster ID and data type if exactly one matching file is found.
        Returns `None` if no matching file is found.

    Raises:
    -------
    ValueError
        If multiple matching files are found for the given cluster ID and data type, indicating an ambiguity that needs to be resolved.

    Example:
    --------
    image_path = get_image_path('/data/images', 23, 'Rainfall')
    if image_path:
        print(f"Found image path: {image_path}")
    else:
        print("No matching image found.")
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

       Parameters:
       -----------
       row : pd.Series
           A Pandas Series representing a row in the DataFrame, where the index of the row is assumed to be the cluster ID (cid).
       image_path : str
           The base directory where the image files are stored.

       Returns:
       --------
       pd.Series
           A Pandas Series containing the file paths for nightlights, population, and rainfall data.
           The paths may be `None` if no matching file is found for the respective data type.
           The Series contains the following indices: 'nightlights_path', 'population_path', 'rainfall_path'.

       Example:
       --------
       # Example usage within a DataFrame apply function
       df[['nightlights_path', 'population_path', 'rainfall_path']] = df.apply(
           lambda row: add_image_paths_for_row(row, '/data/images'), axis=1
       )
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

     Parameters:
     -----------
     geospatial_df : pd.DataFrame
         A DataFrame containing geospatial data, with each row representing a cluster.
     dataset_config : object
         The configuration object containing dataset settings, such as the root directory for AOI data (`AOI_ROOT`).
     country_code : str
         The country code used to specify the folder containing the image tiles for the corresponding AOI.

     Returns:
     --------
     pd.DataFrame
         The updated DataFrame with new columns: 'nightlights_path', 'population_path', and 'rainfall_path'.
         These columns contain the file paths for respective data types, which may be `None` if no matching file is found.

     Example:
     --------
     updated_df = add_geospatial_image_paths(geospatial_df, dataset_config, 'PK')
     if updated_df[['nightlights_path', 'population_path', 'rainfall_path']].isnull().any().any():
         print("Some rows have missing image paths, further inspection required.")

     Notes:
     ------
     - If there are missing paths for any data type, a warning is printed, and those rows are retained for further inspection.
     - This function is useful for ensuring that all necessary geospatial image data is accounted for, aiding in downstream analysis.
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
    """
    Computes the mean and standard deviation of pixel values for a raster image.

    Parameters:
    -----------
    image_path : str
        The file path to the raster image for which the statistics are to be computed.

    Returns:
    --------
    tuple
        A tuple containing the mean and standard deviation of the pixel values in the raster image.
        - mean_value (float): The mean of the pixel values.
        - std_value (float): The standard deviation of the pixel values.

    Example:
    --------
    mean, std = compute_raster_statistics('path/to/raster_image.tif')
    print(f"Mean: {mean}, Standard Deviation: {std}")

    Notes:
    ------
    - This function assumes the raster image has a single channel.
    - The mean and standard deviation are computed over all the pixel values of the raster.
    """
    with rasterio.open(image_path) as src:
        data = src.read(1)  # Assume single channel image
        mean_value = np.mean(data)
        std_value = np.std(data)
    return mean_value, std_value

def add_statistics_to_dataframe(df, data_types):
    """
    Computes the mean and standard deviation of pixel values for a raster image.

    Parameters:
    -----------
    image_path : str
        The file path to the raster image for which the statistics are to be computed.

    Returns:
    --------
    tuple
        A tuple containing the mean and standard deviation of the pixel values in the raster image.
        - mean_value (float): The mean of the pixel values.
        - std_value (float): The standard deviation of the pixel values.

    Example:
    --------
    mean, std = compute_raster_statistics('path/to/raster_image.tif')
    print(f"Mean: {mean}, Standard Deviation: {std}")

    Notes:
    ------
    - This function assumes the raster image has a single channel.
    - The mean and standard deviation are computed over all the pixel values of the raster.
    """
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
    Assigns cluster labels to a geospatial DataFrame by matching cluster IDs with given labels.

    Parameters:
    -----------
    geospatial_df : pandas.DataFrame
        The geospatial DataFrame to which the cluster labels will be assigned.
    cluster_ids : list or array-like
        A list of cluster IDs to match with the corresponding labels.
    cluster_labels : list or array-like
        A list of cluster labels to be assigned to the respective cluster IDs.

    Returns:
    --------
    pandas.DataFrame
        The updated geospatial DataFrame with a new 'cluster_label' column containing the assigned cluster labels.

    Raises:
    -------
    ValueError
        If the lengths of `cluster_ids` and `cluster_labels` are not the same.

    Notes:
    ------
    - If the 'cluster_label' column already exists in the DataFrame, the function will skip the assignment.
    - If any cluster labels are missing after merging, a warning will be printed indicating the number of missing labels.

    Example:
    --------
    updated_df = assign_cluster_labels_by_matching(geospatial_df, cluster_ids, cluster_labels)
    print(updated_df.head())
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
        Computes and normalizes aggregated statistics (mean and standard deviation) for each geospatial data type
        for a specified cluster using the 'cluster_label' column to filter the cluster. Also calculates correlations
        between geospatial data types and survey metrics, and returns vaccination rates for both cluster and non-cluster points.

        Parameters:
        -----------
        geospatial_df_copy : pandas.DataFrame
            A DataFrame containing geospatial data, including a 'cluster_label' column to identify clusters.
        cluster_label : int or str
            The cluster label used to filter the data and perform computations.

        Returns:
        --------
        cluster_stats : dict
            A dictionary containing aggregated and normalized mean and standard deviation statistics for the specified cluster.
        non_cluster_stats : dict
            A dictionary containing aggregated and normalized mean and standard deviation statistics for points outside the cluster.
        cluster_count : int
            The number of points in the specified cluster.
        non_cluster_count : int
            The number of points outside the specified cluster.
        unnormalized_cluster_means : dict
            A dictionary of unnormalized means for display purposes for the cluster.
        unnormalized_non_cluster_means : dict
            A dictionary of unnormalized means for display purposes for the non-cluster.
        survey_means_cluster : list
            A list of mean values for the cluster for survey metrics.
        correlations : dict
            A dictionary containing Pearson correlations between geospatial data types and survey metrics for the cluster.
        cluster_vaccination_rates : list
            A list of raw vaccination rates for points in the cluster.
        non_cluster_vaccination_rates : list
            A list of raw vaccination rates for points outside the cluster.

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