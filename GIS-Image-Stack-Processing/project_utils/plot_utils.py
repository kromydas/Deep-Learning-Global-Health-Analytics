import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from io import BytesIO
from PIL import Image

import seaborn as sns
from collections import Counter
from itertools import combinations

import pyproj
from bokeh.plotting import show
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.models import LinearColorMapper, ColorBar, ColorMapper
from bokeh.io import export_png

from bokeh.plotting import figure
from bokeh.tile_providers import CARTODBPOSITRON
from skgstat import Variogram

def plot_aoi_distribution(dataset, case, title="AOI Distribution"):
    """
    Plots a bar chart showing the number of samples per Area of Interest (AOI) in the dataset.

    Parameters:
    -----------
    dataset : Dataset
        An instance of the MultiChannelGeoTiffDataset, which contains information about the samples and associated AOIs.
    case : str
        A string identifier for the specific case being plotted, used to save the plot to a file.
    title : str, optional (default="AOI Distribution")
        The title of the plot.

    Returns:
    --------
    None
        This function does not return anything. It generates and saves the bar chart as a PNG file.

    Notes:
    ------
    - The function uses a Counter to count the number of samples per AOI in the dataset.
    - The bar chart is saved in the current directory with the filename format "<case>.png".
    - The plot is displayed after being saved, with a fixed y-axis limit of 1000.
    - Ensure that matplotlib is installed and available in your environment.

    Example:
    --------
    plot_aoi_distribution(
        dataset=multi_channel_dataset,
        case='case_1',
        title='AOI Sample Distribution for Case 1'
    )
    """
    # Initialize a counter for AOIs
    aoi_counter = Counter()

    # Iterate over the dataset
    for idx in range(len(dataset)):
        # Retrieve cluster_id and aoi
        _, (cluster_id, aoi, _) = dataset[idx]
        # Increment the AOI counter
        aoi_counter[aoi] += 1

    # Extract AOIs and their counts
    aois = list(aoi_counter.keys())
    counts = list(aoi_counter.values())

    # Create the bar chart
    plt.figure(figsize=(10, 4))
    bars = plt.bar(aois, counts, color='skyblue')

    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.ylim(0, 1000)
    plt.xticks(rotation=45)
    plt.tight_layout()

    file_suffix = ".png"
    file_path = os.path.join('./', f"{case}{file_suffix}")

    plt.savefig(file_path, format='png', dpi=300)

    plt.show()

def plot_training_curves(metrics,
                         title=None,
                         ylabel=None,
                         ylim=None,
                         metric_names=None,
                         colors=None,
                         out_dir=None,
                         case=None):
    """
    Plots training curves for multiple metrics over epochs.

    Parameters:
    -----------
    metrics : list of list of float
        A list of metric values, where each element is a list containing values over epochs.
    title : str, optional
        The title of the plot.
    ylabel : str, optional
        The label for the y-axis.
    ylim : tuple of (float, float), optional
        The limits for the y-axis.
    metric_names : list of str, optional
        Names for each metric to be used in the legend.
    colors : list of str, optional
        Colors to use for each metric line. Should match the number of metrics.
    out_dir : str, optional
        Directory to save the output plot. If not provided, the plot will not be saved.
    case : str, optional
        A string identifier used to generate the filename when saving the plot.

    Returns:
    --------
    None
        This function does not return anything. It generates and optionally saves a plot of the training metrics.

    Notes:
    ------
    - The x-axis represents epochs, starting from 0 to the number of completed epochs.
    - The function creates a line plot for each metric in the list of metrics.
    - If `out_dir` is provided and does not exist, it will be created.
    - The plot will be saved as a PNG file with the name format "<case>.png" in the specified directory.
    - Ensure that matplotlib is installed and available in your environment for the function to work properly.

    Example:
    --------
    metrics = [[0.8, 0.85, 0.9, 0.92], [0.75, 0.8, 0.85, 0.88]]
    plot_training_curves(
        metrics=metrics,
        title='Training Curves',
        ylabel='Accuracy',
        ylim=(0.5, 1.0),
        metric_names=['Training Accuracy', 'Validation Accuracy'],
        colors=['blue', 'orange'],
        out_dir='plots',
        case='training_case_1'
    )
    """
    # Determine the actual number of completed epochs based on the metrics
    actual_epochs = len(metrics[0])  # Assumes all metrics have the same length

    fig, ax = plt.subplots(figsize=(15, 4))

    for idx, metric in enumerate(metrics):
        ax.plot(range(actual_epochs), metric, color=colors[idx], label=metric_names[idx])

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, actual_epochs - 1])

    if ylim:
        plt.ylim(ylim)

    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(5.0))

    plt.grid(True)
    plt.legend()

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the file name for saving
    file_suffix = ".png"
    file_path = os.path.join(out_dir, f"{case}{file_suffix}") if out_dir else f"{case}{file_suffix}"

    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")

    plt.show()
    plt.close()

def plot_predictions(predictions,
                     actuals,
                     out_dir,
                     case=None,
                     add_grid=False,
                     selected_targets=None,
                     title_string=None):
    """
       Plots scatter plots of predictions vs. actuals for multiple targets.

       Parameters:
       -----------
       predictions : array-like
           Predicted values, with shape (num_samples, num_targets).
       actuals : array-like
           Actual values, with shape (num_samples, num_targets). Must match the shape of predictions.
       out_dir : str
           Directory where the plot will be saved.
       case : str, optional
           A string identifier used to generate the filename when saving the plot.
       add_grid : bool, optional (default=False)
           If True, adds a grid to each subplot.
       selected_targets : list of str, optional
           Names for each target output to be used in the plot titles. If not provided, default names are used.
       title_string : str, optional
           Additional string to be appended to each plot title.

       Returns:
       --------
       None
           This function does not return anything. It generates and optionally saves the plots.

       Notes:
       ------
       - The function plots a scatter plot of actual vs. predicted values for each target.
       - The output plot is saved as a PNG file in the specified output directory.
       - Ensure that matplotlib and numpy are installed and available in your environment.

       Example:
       --------
       predictions = np.random.rand(100, 3)
       actuals = np.random.rand(100, 3)
       plot_predictions(
           predictions=predictions,
           actuals=actuals,
           out_dir='plots',
           case='experiment_1',
           add_grid=True,
           selected_targets=['Target 1', 'Target 2', 'Target 3'],
           title_string='Experiment Results'
       )
       """
    predictions = np.array(predictions)
    actuals = np.array(actuals)  # Converted to array

    print(f"Converted Predictions shape: {predictions.shape}")
    print(f"Converted Actuals shape: {actuals.shape}")

    # Ensure shapes match
    if predictions.shape != actuals.shape:
        raise ValueError("Predictions and actual values must have the same shape.")

    num_targets = predictions.shape[1]  # Number of target outputs
    if selected_targets is None:
        selected_targets = [f'Target {i + 1}' for i in range(num_targets)]
    elif len(selected_targets) != num_targets:
        raise ValueError("Length of selected_targets must match the number of targets")

    plt.figure(figsize=(16, 6 * num_targets))  # Adjusted figure size dynamically

    for i in range(num_targets):
        # Scatter plot
        ax1 = plt.subplot(num_targets, 1, i + 1)

        ax1.tick_params(axis='both', labelsize=14)
        plt.scatter(actuals[:, i], predictions[:, i], color='blue', alpha=0.4)
        correlation_coefficient = np.corrcoef(actuals[:, i], predictions[:, i])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.2f}',
                 ha='left', va='top', transform=ax1.transAxes,
                 fontsize=14, bbox=dict(facecolor='white', alpha=0.4))
        plt.plot([0, 1], [0, 1], 'r-', lw=1)

        plt.title(f'{selected_targets[i]} - Predictions vs Actuals: ' + title_string, fontsize=18)
        plt.xlabel('Actual Values', fontsize=16)
        plt.ylabel('Predicted Values', fontsize=16)
        plt.xlim(-0.05, 1.05)
        if add_grid:
            plt.grid(True, linestyle='-', color='lightgray', alpha=0.4)

    plt.tight_layout()

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the file name for saving
    file_suffix = ".png"
    file_path = os.path.join(out_dir, f"{case}{file_suffix}") if out_dir else f"{case}{file_suffix}"

    # Save the plot
    if out_dir:
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")

    plt.show()
    plt.close()


def plot_clusters(projected_features,
                  cluster_ids,
                  cluster_colors,
                  vaccination_rates,
                  num_components=2,
                  out_dir=None,
                  case=None,
                  plot_title=None,
                  annotate_points=False,
                  font_size=8,
                  use_distinct_colors=False,
                  save_to_disk=True):
    """
        Plots the clusters in projected feature space, using different components of the feature representation.

        Parameters:
        -----------
        projected_features : ndarray
            A 2D array where rows represent samples and columns represent the projected feature dimensions.
        cluster_ids : list
            List of cluster identifiers, which are either DHS cluster IDs or labels assigned by a clustering algorithm.
        cluster_colors : list
            List of colors to use for each cluster when plotting. Should have enough colors for the distinct clusters.
        vaccination_rates : list
            List of vaccination rates for each cluster. Each element is expected to be a numeric value representing the rate.
        num_components : int, optional (default=2)
            Number of components to plot. Must be between 2 and the number of columns in `projected_features`.
        out_dir : str, optional
            Directory to save the output plots. If not provided, plots will not be saved to disk.
        case : str, optional
            A string identifier for the specific case being plotted, used to generate file names for saving.
        plot_title : str, optional
            Title of the plot.
        annotate_points : bool, optional (default=False)
            If True, annotates each point with its cluster ID.
        font_size : int, optional (default=8)
            Font size to use for point annotations, if `annotate_points` is True.
        use_distinct_colors : bool, optional (default=False)
            If True, attempts to use distinct colors for each cluster based on `cluster_colors`. If there are more clusters
            than colors available, falls back to a default color.
        save_to_disk : bool, optional (default=True)
            If True, saves the plots to disk in the specified `out_dir`.

        Returns:
        --------
        None
            This function does not return anything. It generates and optionally saves cluster plots for visualization.

        Notes:
        ------
        - The function uses Bokeh to create scatter plots for clusters based on the specified components.
        - Each point represents a cluster, colored according to `cluster_colors`, and optionally annotated with its cluster ID.
        - If `use_distinct_colors` is True and sufficient colors are provided, each cluster is given a distinct color.
        - The plot includes a hover tooltip to display the vaccination rate for each cluster.
        - Plots can be saved to disk if `save_to_disk` is True, using the provided output directory and case identifier.
        - Ensure that Bokeh and numpy are installed and available in your environment.

        Example:
        --------
        plot_clusters(
            projected_features=np.random.rand(100, 3),
            cluster_ids=[1, 2, 3, 4, 5] * 20,
            cluster_colors=['red', 'green', 'blue', 'orange', 'purple'],
            vaccination_rates=np.random.rand(100),
            num_components=2,
            out_dir='plots',
            case='example_case',
            plot_title='Cluster Visualization',
            annotate_points=True,
            font_size=10,
            use_distinct_colors=True,
            save_to_disk=True
        )
        """
    n_components = projected_features.shape[1]

    if num_components < 2 or num_components > n_components:
        raise ValueError(f"Invalid number of components: {num_components}. It must be between 2 and {n_components}.")

    if num_components == 2:
        component_pairs = [(0, 1)]  # Just one plot for the first two components
    else:
        component_pairs = list(combinations(range(num_components), 2))  # All pairs of components

    # Ensure distinct colors are used
    if use_distinct_colors and len(set(cluster_ids)) <= len(cluster_colors):
        cluster_sizes = {cluster: list(cluster_ids).count(cluster) for cluster in set(cluster_ids)}
        sorted_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)
        cluster_color_mapping = {cluster: cluster_colors[i % len(cluster_colors)] for i, cluster in
                                 enumerate(sorted_clusters)}
        color_column = [cluster_color_mapping[cluster] for cluster in cluster_ids]
    else:
        color_column = ['navy' for _ in cluster_ids]  # Fallback if distinct colors aren't used or too many clusters

    for (i, j) in component_pairs:
        # Prepare data source for Bokeh
        source = ColumnDataSource(data={
            'x': projected_features[:, i],
            'y': projected_features[:, j],
            'cluster_id': [str(x) for x in cluster_ids],
            'vaccination_rate': [f"{rate:.2%}" for rate in vaccination_rates],
            'color': color_column
        })

        p = figure(title=plot_title, x_axis_label=f'Component {i + 1}', y_axis_label=f'Component {j + 1}', width=800,
                   height=800)
        p.title.text_font_size = '14pt'

        hover = HoverTool(tooltips=[("Vaccination Rate", "@vaccination_rate"),
                                    ("Cluster ID", "@cluster_id")])

        p.add_tools(hover)

        # Plot using the color column from the source
        p.scatter('x', 'y', source=source, color='color', size=9, alpha=0.5)

        if annotate_points:
            labels = LabelSet(x='x', y='y', text='cluster_id', level='glyph', x_offset=5, y_offset=5, source=source,
                              text_font_size=f"{font_size}pt")
            p.add_layout(labels)

        if save_to_disk:

            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)

            file_suffix = f"_comp_{i + 1}_{j + 1}.png"
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                temp_png_path = tmpfile.name
                export_png(p, filename=temp_png_path)
                with open(temp_png_path, 'rb') as file:
                    buf = BytesIO(file.read())
                buf.seek(0)

            output_path = os.path.join(out_dir, f"{case}{file_suffix}")
            with open(output_path, 'wb') as out_file:
                out_file.write(buf.getvalue())
            print(f"Plot saved to {output_path}")

            os.remove(temp_png_path)

        show(p)

    return None


def plot_cluster_hist_comparison(country_code,
                                 cluster_stats,
                                 non_cluster_stats,
                                 cluster_label,
                                 cluster_count,
                                 non_cluster_count,
                                 unnormalized_cluster_means,
                                 unnormalized_non_cluster_means,
                                 survey_means_cluster,
                                 correlations,
                                 cluster_vaccination_rates,
                                 non_cluster_vaccination_rates,
                                 cluster_color,
                                 stats_normalized=True,
                                 out_dir=None,
                                 case=None,
                                 plot_title=None,
                                 save_to_disk=True,
                                 plot_values=False):
    """
        Plots a bar chart comparing the normalized statistics for a specific cluster against non-cluster statistics,
        along with the distribution of vaccination rates for both the cluster and non-cluster groups.

        Parameters:
        -----------
        country_code : str
            The code of the country being analyzed, used in the title of the plot.
        cluster_stats : dict
            Dictionary containing statistics for the cluster (e.g., means of certain metrics).
        non_cluster_stats : dict
            Dictionary containing statistics for non-cluster data points.
        cluster_label : int
            Label or identifier of the cluster being analyzed.
        cluster_count : int
            Number of points in the cluster.
        non_cluster_count : int
            Number of points outside the cluster.
        unnormalized_cluster_means : dict
            Dictionary of unnormalized mean values for the cluster metrics.
        unnormalized_non_cluster_means : dict
            Dictionary of unnormalized mean values for the non-cluster metrics.
        survey_means_cluster : list
            List of mean values for specific survey metrics for the cluster (e.g., vaccination, access to fresh water).
        correlations : dict
            Dictionary containing correlation values between geospatial data types and survey metrics.
        cluster_vaccination_rates : list
            List of vaccination rates for points within the cluster.
        non_cluster_vaccination_rates : list
            List of vaccination rates for points outside the cluster.
        cluster_color : str
            Color used for the cluster bars and vaccination distribution in the plot.
        stats_normalized : bool, optional (default=True)
            If True, plots normalized mean values; otherwise, plots raw mean values.
        out_dir : str, optional
            Directory where the output plot will be saved. If not provided, the plot will not be saved.
        case : str, optional
            A string identifier used to generate the filename when saving the plot.
        plot_title : str, optional
            Title of the plot.
        save_to_disk : bool, optional (default=True)
            If True, saves the plot to disk in the specified output directory.
        plot_values : bool, optional (default=False)
            If True, displays the unnormalized mean values inside the bars of the bar chart.

        Returns:
        --------
        None
            This function does not return anything. It generates and optionally saves a bar chart and histogram plot.

        Notes:
        ------
        - The first subplot displays a bar chart comparing the means of the cluster and non-cluster metrics.
        - The second subplot displays a histogram comparing the vaccination rate distribution between cluster and non-cluster.
        - The bar chart can include additional text displaying survey metric means for the cluster.
        - Correlation values for each metric are included in the x-axis labels.
        - The function uses seaborn for histogram plotting and matplotlib for bar chart plotting.

        Example:
        --------
        plot_comparison(
            country_code='PK',
            cluster_stats={'nightlights_mean': {'mean': 0.4}, 'rainfall_mean': {'mean': 0.6}},
            non_cluster_stats={'nightlights_mean': {'mean': 0.3}, 'rainfall_mean': {'mean': 0.5}},
            cluster_label=1,
            cluster_count=100,
            non_cluster_count=200,
            unnormalized_cluster_means={'nightlights_mean': 0.45, 'rainfall_mean': 0.65},
            unnormalized_non_cluster_means={'nightlights_mean': 0.35, 'rainfall_mean': 0.55},
            survey_means_cluster=[0.7, 0.6, 0.8, 0.5, 0.4, 0.3],
            correlations={'nightlights_mean': {'fraction_dpt3_vaccinated': 0.5}, 'rainfall_mean': {'fraction_dpt3_vaccinated': 0.4}},
            cluster_vaccination_rates=[0.7, 0.75, 0.72, 0.68],
            non_cluster_vaccination_rates=[0.6, 0.62, 0.58, 0.65],
            cluster_color='blue',
            stats_normalized=True,
            out_dir='plots',
            case='pk_cluster_comparison',
            plot_title='Cluster vs Non-Cluster Statistics for PK',
            save_to_disk=True,
            plot_values=True
        )
        """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

    categories = list(cluster_stats.keys())
    x = np.arange(len(categories))
    width = 0.35  # Bar width

    # Bar chart on the top (cluster vs non-cluster)
    if stats_normalized:
        y_label = 'Normalized Mean'
        cluster_means = [cluster_stats[cat]['mean'] for cat in categories]
        non_cluster_means = [non_cluster_stats[cat]['mean'] for cat in categories]
    else:
        y_label = 'Mean'
        cluster_means = [unnormalized_cluster_means[cat] for cat in categories]
        non_cluster_means = [unnormalized_non_cluster_means[cat] for cat in categories]

    non_cluster_color = 'lightgrey'

    # Plot bar chart in the first subplot (top)
    bars1 = axs[0].bar(x - width / 2, cluster_means, width, label=f'Cluster {cluster_label} Mean (n={cluster_count})',
                       alpha=0.5, color=cluster_color)
    bars2 = axs[0].bar(x + width / 2, non_cluster_means, width, label=f'Non-Cluster Mean (n={non_cluster_count})',
                       alpha=0.5, color=non_cluster_color)

    if plot_values:
        # Add labels with the unnormalized means inside the bars near the top
        for i, (bar, unnormalized_mean) in enumerate(zip(bars1, unnormalized_cluster_means.values())):
            posx = bar.get_x() + bar.get_width() / 2
            posy = bar.get_height() - 0.08
            axs[0].text(posx, posy, f'{unnormalized_mean:.2f}', ha='center', va='bottom', color='black')

    # Display survey metrics in the upper left corner of the plot
    survey_metrics = ['Cluster Mean Fraction Vaccinated', 'Cluster Mean Fraction with Fresh Water',
                      'Cluster Mean Fraction with Electricity', 'Cluster Mean Fraction with Radio',
                      'Cluster Mean Fraction with TV', 'Cluster Mean Wealth Index']
    for i, (metric, mean_value) in enumerate(zip(survey_metrics, survey_means_cluster)):
        axs[0].text(0.05, 0.95 - (i * 0.05), f'{metric}: {mean_value:.2f}',
                    ha='left', va='top', transform=axs[0].transAxes, fontsize=10, color=cluster_color)

    # Modify the x-axis labels to include correlation values for vaccination
    categories_with_corr = [
        f"{cat.replace('_mean', '').capitalize()} [Vacc. Corr: {correlations[cat]['fraction_dpt3_vaccinated']:.2f}]" for
        cat in categories]
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(categories_with_corr, rotation=0, ha='center')
    axs[0].set_ylabel(y_label)
    axs[0].set_title(plot_title)
    axs[0].legend()

    # Set y-axis limit for the bar chart
    if stats_normalized:
        axs[0].set_ylim(0, 1.4)
    else:
        axs[0].autoscale()

    # Plot the vaccination distribution below the bar chart
    if stats_normalized:
        y_label = 'Relative Freq.'
        sns.histplot(cluster_vaccination_rates, bins=10, kde=True, alpha=0.5, color=cluster_color, ax=axs[1],
                     label='Cluster Vaccination Rates', stat='probability')
        sns.histplot(non_cluster_vaccination_rates, bins=10, kde=True, alpha=0.5, color=non_cluster_color, ax=axs[1],
                     label='Non-Cluster Vaccination Rates', stat='probability')
    else:
        y_label = 'Counts'
        sns.histplot(cluster_vaccination_rates, bins=10, kde=True, alpha=0.5, color=cluster_color, ax=axs[1],
                     label='Cluster Vaccination Rates')
        sns.histplot(non_cluster_vaccination_rates, bins=10, kde=True, alpha=0.5, color=non_cluster_color, ax=axs[1],
                     label='Non-Cluster Vaccination Rates')
        axs[1].set_ylim(0, 500)
        # axs[0].autoscale()

    axs[1].set_xlabel('Vaccination Rate')
    axs[1].set_ylabel(y_label)
    #axs[1].set_title(f'{country_code}: Vaccination Distribution')
    if non_cluster_count > 0:
        axs[1].legend()

    # Save to disk if needed
    if save_to_disk:
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        file_suffix = ".png"
        file_path = os.path.join(out_dir, f"{case}{file_suffix}") if out_dir else f"{case}{file_suffix}"
        plt.savefig(file_path, format='png')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_cluster_violin_comparison(country_code,
                                   cluster_vaccination_rates,
                                   non_cluster_vaccination_rates,
                                   cluster_labels,
                                   cluster_sizes,
                                   cluster_colors,
                                   n_clusters=None,
                                   feature_string=None,
                                   model_string=None,
                                   plot_title=None,
                                   out_dir=None,
                                   case=None,
                                   save_to_disk=True):
    """
    Plots a single violin plot comparing vaccination rate distributions for all clusters in one figure.
    Ensures individual violin plots maintain the same width as if there were 9 clusters, aligned to the left.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os

    # Generate default case and plot_title if not provided
    if case is None and n_clusters is not None and feature_string is not None:
        case = f"{country_code}_{n_clusters}Cluster_Compare_{feature_string}"

    if plot_title is None and model_string is not None and feature_string is not None:
        plot_title = f"{country_code}: Cluster Comparison ({model_string}), Features: {feature_string}"

    # Prepare data for seaborn violin plot
    data = []
    labels = []
    colors = []
    means = []

    # Add each cluster's vaccination rates to the dataset
    for i, rates in enumerate(cluster_vaccination_rates):
        if isinstance(rates, float):
            rates = [rates]
        data.extend(rates)
        labels.extend([f"Cluster {cluster_labels[i]} (n={cluster_sizes[i]})"] * len(rates))
        colors.extend([cluster_colors[i]] * len(rates))
        means.append(sum(rates) / len(rates) if len(rates) > 0 else 0)

    # Add non-cluster vaccination rates to the dataset
    if non_cluster_vaccination_rates:
        if isinstance(non_cluster_vaccination_rates, float):
            non_cluster_vaccination_rates = [non_cluster_vaccination_rates]
        data.extend(non_cluster_vaccination_rates)
        labels.extend([f"Non-Cluster (n={len(non_cluster_vaccination_rates)})"] * len(non_cluster_vaccination_rates))
        colors.extend(["gray"] * len(non_cluster_vaccination_rates))
        means.append(sum(non_cluster_vaccination_rates) / len(non_cluster_vaccination_rates) if len(
            non_cluster_vaccination_rates) > 0 else 0)

    # Create a DataFrame for plotting
    df = pd.DataFrame({"Vaccination Rate": data, "Group": labels, "Color": colors})

    # Prepare fixed positions for up to 9 total violins
    max_clusters = 9
    num_groups = len(cluster_labels) + (1 if non_cluster_vaccination_rates else 0)
    fixed_positions = np.linspace(0, max_clusters - 1, max_clusters)  # Fixed slots for 9 positions
    positions = fixed_positions[:num_groups]  # Use only the needed positions for the current groups

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 5))

    # Generate group labels and colors
    order = [f"Cluster {cluster_labels[i]} (n={cluster_sizes[i]})" for i in range(len(cluster_labels))]
    if non_cluster_vaccination_rates:
        order.append(f"Non-Cluster (n={len(non_cluster_vaccination_rates)})")

    all_colors = cluster_colors + ["gray"]  # Combine cluster colors with gray for Non-Cluster

    # Create a custom palette mapping each label to its color
    palette = {label: color for label, color in zip(order, all_colors)}

    # Create the violin plot with custom positions
    sns.violinplot(data=df,
                   x="Group",
                   y="Vaccination Rate",
                   hue="Group",
                   order=order,
                   palette=palette,
                   inner="points",
                   linewidth=1,
                   alpha=0.7,
                   density_norm="width",
                   width=0.8,
                   ax=ax)

    # Adjust x-axis limits to ensure violins are positioned as if there were always 9
    ax.set_xlim(-0.5, max_clusters - 0.5)

    # Adjust y-axis limits to provide ample space for tails
    ax.set_ylim(-0.3, 1.3)  # Increased space for tails without clipping

    # Get the max y-axis limit
    max_y_limit = ax.get_ylim()[1]

    # Add mean vaccination rates as text above each violin plot at 0.92 * max_y_limit
    for i, mean in enumerate(means):
        ax.text(positions[i], 0.92 * max_y_limit, f"{mean:.2f}", ha="center", va="bottom", fontsize=14, color="black")

    # Dynamically place sample count labels directly below each violin plot
    x_labels = [f"n={cluster_sizes[i]}" for i in range(len(cluster_sizes))]
    if non_cluster_vaccination_rates:
        x_labels.append(f"n={len(non_cluster_vaccination_rates)}")
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=14)

    # Set other plot labels
    plt.yticks(fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Vaccination Rate", fontsize=14)
    ax.set_title(plot_title or f"{country_code}: Vaccination Rate Distributions by Cluster", fontsize=12, pad=10)

    # Save the plot if required
    if save_to_disk:
        if out_dir is None:
            out_dir = "plots"  # Default directory if out_dir is not provided
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        file_suffix = ".png"
        file_path = os.path.join(out_dir, f"{case}{file_suffix}") if out_dir else f"{case}{file_suffix}"
        plt.tight_layout()
        plt.savefig(file_path, format="png")

    # Show the plot
    plt.show()


def wgs84_to_mercator(lon, lat):
    """
    Converts geographic coordinates (longitude, latitude) in WGS84 to Mercator projection coordinates.

    Parameters:
    -----------
    lon : float or ndarray
        Longitude in degrees, can be a single value or an array of values.
    lat : float or ndarray
        Latitude in degrees, can be a single value or an array of values.

    Returns:
    --------
    tuple
        A tuple (x, y) representing the Mercator projection coordinates, where:
        - x is the Easting (in meters)
        - y is the Northing (in meters)

    Notes:
    ------
    - The function uses the spherical Mercator projection formula to convert latitude and longitude to Mercator x, y.
    - It assumes a spherical Earth with radius 6378137 meters.
    - The output coordinates are in meters.

    Example:
    --------
    lon, lat = -73.9857, 40.7488
    x, y = wgs84_to_mercator(lon, lat)
    """
    re = 6378137
    x = lon * (re * np.pi/180.0)
    y = np.log(np.tan((90 + lat) * np.pi/360.0)) * re
    return x, y


def create_geospatial_plot(source,
                           tooltips,
                           color_spec,
                           out_dir=None,
                           case=None,
                           plot_title=None,
                           color_bar=False,
                           color_bar_title=None,
                           alpha=.5,
                           symbol_size=10,
                           plot_width=950,
                           save_to_disk=True):
    """
        Creates a geospatial plot with Mercator projection using Bokeh, allowing for customizable color specifications,
        tooltips, and optional color bar.

        Parameters:
        -----------
        source : ColumnDataSource
            A Bokeh ColumnDataSource containing data for the plot, including 'mercator_x' and 'mercator_y' coordinates.
        tooltips : list of tuples
            Tooltips to display on hover, defined as a list of tuples with field names and descriptions.
        color_spec : ColorMapper, dict, or str
            Specifies the color for points in the plot. Can be:
            - A ColorMapper for color scaling.
            - A dictionary containing a color field and transform.
            - A simple color string (e.g., "navy").
        out_dir : str, optional
            Directory where the output plot image will be saved. If not provided, plot will not be saved.
        case : str, optional
            A string identifier used to generate the filename when saving the plot.
        plot_title : str, optional
            Title for the plot.
        color_bar : bool, optional (default=False)
            If True, adds a color bar to the plot.
        color_bar_title : str, optional
            Title for the color bar, if `color_bar` is True.
        alpha : float, optional (default=0.5)
            Symbol transparency fill value.
        symbol_size : int, optional (default=10)
            Size of the symbols used in the scatter plot.
        plot_width : int, optional (default=950)
            Width of the plot in pixels.
        save_to_disk : bool, optional (default=True)
            If True, saves the plot to a PNG file in the specified `out_dir`.

        Returns:
        --------
        None
            This function does not return anything. It generates and optionally saves the geospatial plot.

        Notes:
        ------
        - The function uses the CARTODBPOSITRON map tile as a base map layer.
        - The plot title, axis labels, and hover tooltips can be customized.
        - If `color_spec` is a ColorMapper or dictionary with a transform, the plot will use a color scale.
        - If a color bar is needed, it will be added to the right of the plot.
        - Uses Bokeh's `export_png` for saving plots to PNG files, and displays the plot in the browser with `show()`.

        Example:
        --------
        create_geospatial_plot(
            source=source,
            tooltips=[("Cluster ID", "@cluster_id"), ("Vaccination Rate", "@vaccination_rate")],
            color_spec="navy",
            out_dir='plots',
            case='example_case',
            plot_title='Geospatial Distribution of Vaccination Rates',
            color_bar=True,
            color_bar_title='Vaccination Rate',
            symbol_size=12,
            plot_width=1000,
            save_to_disk=True
        )
        """
    # Create figure with specified settings
    p = figure(title=plot_title, x_axis_type="mercator", y_axis_type="mercator",
               tools="pan,wheel_zoom,box_zoom,reset",
               width=plot_width, height=800)

    p.add_tile(CARTODBPOSITRON)

    # Set axis labels and font sizes
    p.xaxis.axis_label = 'Longitude'
    p.yaxis.axis_label = 'Latitude'
    p.axis.major_label_text_font_size = "11pt"
    p.title.text_font_size = '14pt'  # Set the title font size
    p.xaxis.axis_label_text_font_size = '14pt'  # Set the font size for x-axis label
    p.yaxis.axis_label_text_font_size = '14pt'  # Set the font size for y-axis label
    p.axis.major_label_text_font_size = "12pt"

    # Initialize color_mapper as None
    color_mapper = None

    # Handle different color specifications
    if isinstance(color_spec, dict) and 'transform' in color_spec and isinstance(color_spec['transform'], ColorMapper):
        color_mapper = color_spec['transform']
        field = color_spec['field']
        p.scatter('mercator_x', 'mercator_y', source=source, fill_alpha=alpha, size=symbol_size,
                  color={'field': field, 'transform': color_mapper})

    elif hasattr(color_spec, 'transform') and isinstance(color_spec.transform, ColorMapper):
        color_mapper = color_spec.transform
        field = color_spec.field
        p.scatter('mercator_x', 'mercator_y', source=source, fill_alpha=alpha, size=symbol_size,
                  color={'field': field, 'transform': color_mapper})

    elif isinstance(color_spec, ColorMapper):
        color_mapper = color_spec
        p.scatter('mercator_x', 'mercator_y', source=source, fill_alpha=alpha, size=symbol_size,
                  color={'field': 'pc_value', 'transform': color_spec})  # Ensure 'pc_value' is valid

    else:
        p.scatter('mercator_x', 'mercator_y', source=source, fill_alpha=alpha, size=symbol_size, color="navy")
        print("Added circles with simple or undefined color specification.")

    # Add hover tool
    hover = HoverTool(tooltips=tooltips)
    p.add_tools(hover)

    # Optionally add a color bar
    if color_bar and color_mapper:
        bar = ColorBar(color_mapper=color_mapper, label_standoff=12, width=10, location=(0, 0),
                       major_label_text_font_size='10pt', title_text_font_size='10pt', title=color_bar_title)
        p.add_layout(bar, 'right')

    # Optionally save the plot to a PNG file
    if save_to_disk:
        # Ensure the output directory exists
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Create the file name for saving
        file_suffix = ".png"
        file_path = os.path.join(out_dir, f"{case}{file_suffix}") if out_dir else f"{case}{file_suffix}"

        # Use a temporary file to save the PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            temp_png_path = tmpfile.name

        # Save plot to the temporary file using Bokeh's export_png
        export_png(p, filename=temp_png_path)

        # Read the PNG file into a BytesIO object if needed
        with open(temp_png_path, 'rb') as file:
            buf = BytesIO(file.read())

        buf.seek(0)  # Reset buffer pointer to the beginning

        # Save the image to the output directory
        with open(file_path, 'wb') as out_file:
            out_file.write(buf.getvalue())
        print(f"Plot saved to {file_path}")

        # Clean up: delete the temporary file
        os.remove(temp_png_path)

    # Show the plot in the browser
    show(p)


def plot_variograms(country_code,
                    geospatial_df,
                    aoi_configurations,
                    out_dir=None,
                    normalize=True,
                    variogram_model='Spherical',
                    n_lags=20,
                    max_lag_km=1500):
    """
       Plots variograms for geospatial features such as vaccination rates, electricity coverage, wealth index, and more, for a given country.
       The variograms provide a visual representation of the spatial correlation of each feature over a specified range.

       Parameters:
       -----------
       country_code : str
           The country code representing the Area of Interest (AOI) for which variograms are generated.
       geospatial_df : pandas.DataFrame
           A DataFrame containing the geospatial data, including columns for latitude ('lat'), longitude ('lon'),
           and feature values such as vaccination rates, electricity, etc.
       aoi_configurations : dict
           Dictionary containing AOI-specific configurations such as latitude and longitude of the CRS origin.
           Expected to contain keys as country codes mapping to CRS details ('crs_lat', 'crs_lon').
       out_dir : str, optional
           Directory where the output variogram plot images will be saved. Default is None, in which case plots will not be saved.
       normalize : bool, optional (default=True)
           If True, normalizes lag distances in the variogram plots. If False, plots lag distance in kilometers.
       variogram_model : str, optional (default='Spherical')
           The model type to use for variogram fitting. Common options include 'Spherical', 'Gaussian', 'Exponential', etc.
       n_lags : int, optional (default=20)
           Number of lags used for variogram calculation.
       max_lag_km : float, optional (default=1500)
           Maximum lag distance in kilometers for the variogram.

       Returns:
       --------
       None
           This function generates variogram plots for different features, optionally saves them as images, and displays them.

       Notes:
       ------
       - The function calculates variograms for various geospatial features including:
         - Vaccination rates ('fraction_dpt3_vaccinated')
         - Wealth index ('mean_wealth_index')
         - Electricity coverage ('fraction_with_electricity')
         - Radio ownership ('fraction_with_radio')
         - Television ownership ('fraction_with_tv')
       - Fresh water access is excluded from the variogram calculation as not all AOIs have this data.
       - Uses a Lambert Azimuthal Equal-Area (LAEA) projection for transforming coordinates.
       - If any data contains NaN values, they are filtered out prior to variogram calculation.
       - The variograms are plotted using Matplotlib and saved as PNG images.
       - The function combines all variogram images into a single stacked image.

       Example:
       --------
       plot_variograms(
           country_code='PK',
           geospatial_df=df,
           aoi_configurations={'PK': {'crs_lat': 30.3753, 'crs_lon': 69.3451}},
           out_dir='output/variograms',
           normalize=True,
           variogram_model='Spherical',
           n_lags=20,
           max_lag_km=1500
       )
       """
    crs_lat = aoi_configurations[country_code]['crs_lat']
    crs_lon = aoi_configurations[country_code]['crs_lon']

    # print(f"Country: {country_code}, CRS lat: {crs_lat}, lon: {crs_lon}")

    x_axis_label = 'Normalized Lag Distance' if normalize else 'Lag Distance [km]'

    proj_string = f'+proj=laea +lat_0={crs_lat} +lon_0={crs_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    proj = pyproj.Proj(proj_string)

    # print(f"Projection string: {proj_string}")

    latitudes = geospatial_df['lat'].values
    longitudes = geospatial_df['lon'].values
    projected_coords = np.array([proj(lon, lat) for lon, lat in zip(longitudes, latitudes)])
    projected_coords_km = projected_coords / 1000.0

    # print(f"Shape of projected coordinates (km): {projected_coords_km.shape}")

    # Helper function to check if array is valid for variogram calculation
    def is_valid_data(array, coords, name):
        if len(array) == 0:
            print(f"WARNING: {name} data is empty.")
            return None, None
        if np.isnan(array).any():
            print(f"WARNING: {name} data contains NaN values, filtering them out.")
            valid_idx = ~np.isnan(array)
            array = array[valid_idx]
            coords = coords[valid_idx]
        if len(np.unique(array)) <= 1:
            print(f"WARNING: {name} data has only one unique value after filtering, skipping variogram calculation.")
            return None, None
        return array, coords

    # Vaccinations
    vaccination_rates = geospatial_df['fraction_dpt3_vaccinated'].values
    vaccination_rates, projected_coords_km_vacc = is_valid_data(vaccination_rates, projected_coords_km,
                                                                'Vaccination rates')
    if vaccination_rates is not None:
        V_vaccination = Variogram(projected_coords_km_vacc, vaccination_rates, model=variogram_model, n_lags=n_lags,
                                  normalize=normalize, maxlag=max_lag_km)
    else:
        V_vaccination = None

    # Water
    fraction_with_fresh_water = geospatial_df['fraction_with_fresh_water'].values
    fraction_with_fresh_water, projected_coords_km_water = is_valid_data(fraction_with_fresh_water, projected_coords_km,
                                                                         'Fresh water')
    # if fraction_with_fresh_water is not None:
    #     V_fresh_water = Variogram(projected_coords_km_water, fraction_with_fresh_water, model=variogram_model,
    #                               n_lags=n_lags, normalize=normalize, maxlag=max_lag_km)
    # else:
    #     V_fresh_water = None
    # Exclude fresh water access for now since some AOIs do not have such data
    V_fresh_water = None

    # Wealth index
    mean_wealth_index = geospatial_df['mean_wealth_index'].values
    mean_wealth_index, projected_coords_km_wealth = is_valid_data(mean_wealth_index, projected_coords_km,
                                                                  'Wealth index')
    if mean_wealth_index is not None:
        V_wealth_index = Variogram(projected_coords_km_wealth, mean_wealth_index, model=variogram_model, n_lags=n_lags,
                                   normalize=normalize, maxlag=max_lag_km)
    else:
        V_wealth_index = None

    # Electricity
    fraction_with_electricity = geospatial_df['fraction_with_electricity'].values
    fraction_with_electricity, projected_coords_km_elec = is_valid_data(fraction_with_electricity, projected_coords_km,
                                                                        'Electricity')
    if fraction_with_electricity is not None:
        V_electricity = Variogram(projected_coords_km_elec, fraction_with_electricity, model=variogram_model,
                                  n_lags=n_lags, normalize=normalize, maxlag=max_lag_km)
    else:
        V_electricity = None

    # Radio
    fraction_with_radio = geospatial_df['fraction_with_radio'].values
    fraction_with_radio, projected_coords_km_radio = is_valid_data(fraction_with_radio, projected_coords_km, 'Radio')
    if fraction_with_radio is not None:
        V_radio = Variogram(projected_coords_km_radio, fraction_with_radio, model=variogram_model, n_lags=n_lags,
                            normalize=normalize, maxlag=max_lag_km)
    else:
        V_radio = None

    # Radio
    fraction_with_tv = geospatial_df['fraction_with_tv'].values
    fraction_with_tv, projected_coords_km_tv = is_valid_data(fraction_with_tv, projected_coords_km, 'TV')
    if fraction_with_tv is not None:
        V_tv = Variogram(projected_coords_km_tv, fraction_with_tv, model=variogram_model, n_lags=n_lags,
                         normalize=normalize, maxlag=max_lag_km)
    else:
        V_tv = None

    # List to store buffers
    buffers = []

    # Helper function to create and save plots
    def create_plot(variogram, title, xlabel, ylabel):
        if variogram is None:
            print(f"Skipping plot for {title}, no valid data.")
            return
        buf = BytesIO()
        plt.figure()
        variogram.plot()
        fig = plt.gcf()
        axes = fig.get_axes()
        axes[1].set_title(title)
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        plt.show()
        fig.savefig(buf, format='png')
        buf.seek(0)  # Reset buffer pointer to the beginning
        buffers.append(buf)

    # Generate and save the variograms
    create_plot(V_vaccination, f'{country_code}: Variogram of Vaccination Rates', x_axis_label, 'Semivariance')
    create_plot(V_electricity, f'{country_code}: Variogram of Electricity Coverage', x_axis_label, 'Semivariance')
    create_plot(V_fresh_water, f'{country_code}: Variogram of Fresh Water Access', x_axis_label, 'Semivariance')
    create_plot(V_wealth_index, f'{country_code}: Variogram of Wealth Index', x_axis_label, 'Semivariance')
    create_plot(V_radio, f'{country_code}: Variogram of Radio', x_axis_label, 'Semivariance')
    create_plot(V_tv, f'{country_code}: Variogram of TV', x_axis_label, 'Semivariance')

    # Combine all images
    images = [Image.open(buf) for buf in buffers]

    # Get dimensions of the images
    widths, heights = zip(*(image.size for image in images))

    # Calculate the total height for stacking images
    total_height = sum(heights)
    max_width = max(widths)

    # Create a new blank image with enough height to stack all images
    combined_image = Image.new('RGB', (max_width, total_height))

    # Paste the images into the combined image
    y_offset = 0
    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.height

    # Ensure the output directory exists
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the file name for saving
    file_suffix = ".png"
    # file_path = os.path.join(out_dir, f"{case}{file_suffix}") if out_dir else f"{case}{file_suffix}"

    file_path = f'{out_dir}/Normalized/' if normalize else out_dir
    combined_image.save(f'{file_path}/{country_code}_variograms{"_normalized" if normalize else ""}.png')

    # Close the memory buffers
    for buf in buffers:
        buf.close()

def display_rgb_images(data_loader, cluster_id_list=None, cols=4, max_rows=10):
    """
    Display RGB images in a grid format. Optionally filter by cluster IDs.

    Args:
        data_loader: PyTorch DataLoader object.
        cluster_id_list: Optional list of cluster IDs to filter. If None, display all images.
        cols: Number of columns in the grid.
        max_rows: Maximum number of rows to display in the grid.
    """
    # Initialize a list to store filtered images and metadata
    filtered_images = []
    filtered_titles = []

    # Filter images by cluster IDs
    for images, cluster_info in data_loader:
        batch_size = images.shape[0]

        for i in range(batch_size):
            cluster_id = cluster_info[0][i].item()

            # Add only images that match the cluster ID filter
            if cluster_id_list and cluster_id not in cluster_id_list:
                continue

            rgb_image = images[i][:3].permute(1, 2, 0).numpy()
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            vaccination_rate = cluster_info[2][i][0].item()

            filtered_images.append(rgb_image)
            filtered_titles.append(f"Cluster ID: {cluster_id}\nVaccination Rate: {vaccination_rate:.2f}")

            # Stop once we've collected enough images for the grid
            if len(filtered_images) >= max_rows * cols:
                break

        if len(filtered_images) >= max_rows * cols:
            break

    # Adjust the number of rows dynamically, but cap it at max_rows
    num_images = len(filtered_images)
    rows = min((num_images + cols - 1) // cols, max_rows)

    # Dynamically adjust figsize based on rows and cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for idx, (image, title) in enumerate(zip(filtered_images, filtered_titles)):
        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_title(title, fontsize=10)

    # Turn off any unused axes
    for i in range(len(filtered_images), len(axes)):
        axes[i].axis('off')

    # Use constrained_layout for better spacing control
    fig.set_constrained_layout(True)
    plt.show()

def display_ir_band(data_loader, band_index, cluster_id_list=None, cols=4, max_rows=10):
    """
    Display a single IR band in grayscale. Optionally filter by cluster IDs.

    Args:
        data_loader: PyTorch DataLoader object.
        band_index: Index of the IR band to display (0-based index relative to all 6 bands, e.g., 3, 4, or 5).
        cluster_id_list: Optional list of cluster IDs to filter. If None, display all images.
        cols: Number of columns in the grid.
        max_rows: Maximum number of rows to display in the grid.
    """
    if band_index < 3 or band_index > 5:
        raise ValueError("Invalid band_index. Must be 3, 4, or 5 for IR bands.")

    # Initialize a list to store filtered images and metadata
    filtered_images = []
    filtered_titles = []

    # Filter images by cluster IDs
    for images, cluster_info in data_loader:
        batch_size = images.shape[0]

        for i in range(batch_size):
            cluster_id = cluster_info[0][i].item()

            # Add only images that match the cluster ID filter
            if cluster_id_list and cluster_id not in cluster_id_list:
                continue

            ir_band = images[i][band_index, :, :].numpy()
            ir_band_normalized = (ir_band - ir_band.min()) / (ir_band.max() - ir_band.min())
            vaccination_rate = cluster_info[2][i][0].item()

            filtered_images.append(ir_band_normalized)
            filtered_titles.append(f"Cluster ID: {cluster_id}\nVaccination Rate: {vaccination_rate:.2f}")

            # Stop once we've collected enough images for the grid
            if len(filtered_images) >= max_rows * cols:
                break

        if len(filtered_images) >= max_rows * cols:
            break

    # Adjust the number of rows dynamically, but cap it at max_rows
    num_images = len(filtered_images)
    rows = min((num_images + cols - 1) // cols, max_rows)

    # Dynamically adjust figsize based on rows and cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for idx, (image, title) in enumerate(zip(filtered_images, filtered_titles)):
        axes[idx].imshow(image, cmap="gray")
        axes[idx].axis('off')
        axes[idx].set_title(title, fontsize=10)

    # Turn off any unused axes
    for i in range(len(filtered_images), len(axes)):
        axes[i].axis('off')

    # Use constrained_layout for better spacing control
    fig.set_constrained_layout(True)
    plt.show()
