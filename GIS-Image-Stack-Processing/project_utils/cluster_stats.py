
def compute_normalized_weighted_means_with_deltas_from_df(geospatial_df_copy, cluster_data):
    """
    Compute normalized weighted means, percentage contributions, and percent below/above AOI mean using DataFrame inputs.

    Args:
        geospatial_df_copy (pd.DataFrame): DataFrame containing sample-level data, including vaccination rates.
        cluster_data (pd.DataFrame): DataFrame containing cluster-level aggregated data with 'mean' and 'count'.

    Returns:
        dict: A dictionary containing normalized weighted means, percentage contributions,
              and percent below/above the AOI mean.
    """
    # Compute AOI mean vaccination rate and total number of samples
    aoi_mean = geospatial_df_copy['fraction_dpt3_vaccinated'].mean()
    aoi_total_samples = len(geospatial_df_copy)

    # Separate low and high clusters based on AOI mean (high includes equality)
    low_clusters = cluster_data[cluster_data['mean'] < aoi_mean]
    high_clusters = cluster_data[cluster_data['mean'] >= aoi_mean]

    # Compute weighted mean for low clusters
    if not low_clusters.empty:
        total_low_samples = low_clusters['count'].sum()
        weighted_low_rate = (low_clusters['mean'] * low_clusters['count']).sum() / total_low_samples
    else:
        weighted_low_rate = 0
        total_low_samples = 0

    # Compute weighted mean for high clusters
    if not high_clusters.empty:
        total_high_samples = high_clusters['count'].sum()
        weighted_high_rate = (high_clusters['mean'] * high_clusters['count']).sum() / total_high_samples
    else:
        weighted_high_rate = 0
        total_high_samples = 0

    # Normalize by AOI mean
    normalized_low_rate = weighted_low_rate / aoi_mean if aoi_mean > 0 else 0
    normalized_high_rate = weighted_high_rate / aoi_mean if aoi_mean > 0 else 0

    # Compute percentage of samples below and above AOI mean
    percent_below = (total_low_samples / aoi_total_samples) * 100 if aoi_total_samples > 0 else 0
    percent_above = (total_high_samples / aoi_total_samples) * 100 if aoi_total_samples > 0 else 0

    # Compute percent below and above AOI mean for normalized weighted means
    delta_below = (1 - normalized_low_rate) * 100 if normalized_low_rate <= 1 else 0
    delta_above = (normalized_high_rate - 1) * 100 if normalized_high_rate >= 1 else 0

    # Return results
    return {
        "AOI Mean": aoi_mean,
        "Normalized Weighted Mean for Low Clusters": normalized_low_rate,
        "Normalized Weighted Mean for High Clusters": normalized_high_rate,
        "Percentage Samples Below AOI Mean": percent_below,
        "Percentage Samples Above AOI Mean": percent_above,
        "Percent Weighted Mean Below AOI Mean (Delta)": delta_below,
        "Percent Weighted Mean Above AOI Mean (Delta)": delta_above
    }

def compute_weighted_skewness_from_df(geospatial_df_copy, sorted_clusters, compute_cluster_aggregated_statistics_and_correlations):
    """
    Compute the weighted skewness metric using all sample points in a DataFrame.

    Args:
        geospatial_df_copy (pd.DataFrame): DataFrame containing the sample-level vaccination data.
        sorted_clusters (list): List of cluster labels in sorted order.
        compute_cluster_aggregated_statistics_and_correlations (function): Function to compute cluster statistics.

    Returns:
        dict: A dictionary containing weighted skewness metrics and sample percentages.
    """
    import numpy as np
    from scipy.stats import skew

    # Initialize storage for low and high cluster data
    low_clusters = []
    high_clusters = []

    # Loop through clusters to get detailed sample data
    for cluster_label in sorted_clusters:
        # Extract vaccination rates for the current cluster
        (_, _, cluster_count, _, _, _, _, _,
         cluster_vaccination_rates, _) = compute_cluster_aggregated_statistics_and_correlations(
            geospatial_df_copy, cluster_label=cluster_label
        )

        # Skip empty clusters
        if cluster_count == 0:
            continue

        # Compute skewness for the current cluster
        std_dev = np.std(cluster_vaccination_rates)
        if len(cluster_vaccination_rates) < 3 or std_dev == 0:  # Handle small or uniform clusters
            cluster_skewness = 0
        else:
            cluster_skewness = skew(cluster_vaccination_rates, bias=False)

        # Classify the cluster as low or high based on the cluster mean
        cluster_mean = np.mean(cluster_vaccination_rates)
        if cluster_mean < np.mean(geospatial_df_copy['fraction_dpt3_vaccinated']):  # Compare with AOI mean
            low_clusters.append((cluster_skewness, cluster_count))
        else:
            high_clusters.append((cluster_skewness, cluster_count))

    # Compute weighted skewness for low clusters
    if low_clusters:
        total_low_samples = sum(count for _, count in low_clusters)
        low_skewness_values = [
            skewness * count for skewness, count in low_clusters
        ]
        weighted_low_skewness = sum(low_skewness_values) / total_low_samples
    else:
        weighted_low_skewness = 0
        total_low_samples = 0

    # Compute weighted skewness for high clusters
    if high_clusters:
        total_high_samples = sum(count for _, count in high_clusters)
        high_skewness_values = [
            skewness * count for skewness, count in high_clusters
        ]
        weighted_high_skewness = sum(high_skewness_values) / total_high_samples
    else:
        weighted_high_skewness = 0
        total_high_samples = 0

    # Compute percentage of samples below and above AOI mean
    total_samples = len(geospatial_df_copy)
    percent_below = (total_low_samples / total_samples) * 100 if total_samples > 0 else 0
    percent_above = (total_high_samples / total_samples) * 100 if total_samples > 0 else 0

    # Return results
    return {
        "Weighted Skewness for Low Clusters": weighted_low_skewness,
        "Weighted Skewness for High Clusters": weighted_high_skewness,
        "Percentage Below AOI Mean": percent_below,
        "Percentage Above AOI Mean": percent_above
    }
