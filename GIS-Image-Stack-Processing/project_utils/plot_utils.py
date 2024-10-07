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
#from bokeh.plotting import figure, show, #output_file, save
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.models import LinearColorMapper, ColorBar, ColorMapper
# from bokeh.models import CheckboxGroup, CustomJS
from bokeh.io import export_png, export_svg
# from bokeh.transform import linear_cmap

from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, CARTODBPOSITRON #ESRI_IMAGERY
from skgstat import Variogram

def plot_aoi_distribution(dataset, case, title="AOI Distribution"):
    """
    Plots a bar chart showing the number of samples per AOI in the dataset.

    Args:
        dataset (Dataset): An instance of the MultiChannelGeoTiffDataset.
        title (str): The title of the plot.
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

    # Save the plot if save_path is provided
    if out_dir:
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
                     plot_hist=False,
                     title_string=None,
                     save_path=None):
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
        if plot_hist:
            ax1 = plt.subplot(num_targets, 2, 2 * i + 1)
        else:
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

        # Histogram of errors
        if plot_hist:
            ax2 = plt.subplot(num_targets, 2, 2 * i + 2)
            errors = predictions[:, i] - actuals[:, i]
            plt.hist(errors, bins=30, color='green', alpha=0.5)
            plt.title(f'{selected_targets[i]} - Error Histogram')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.xlim(-1, 1)
            if add_grid:
                plt.grid(True, linestyle='-', color='lightgray', alpha=0.4)

    plt.tight_layout()

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the file name for saving
    file_suffix = ".png"
    file_path = os.path.join(out_dir, f"{case}{file_suffix}") if out_dir else f"{case}{file_suffix}"

    # Save the plot if save_path is provided
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
                  n_clusters=None,
                  save_to_disk=True):
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

        # Add hover tool with vaccination rate
        #         hover = HoverTool(tooltips=[("Cluster",          "@cluster_id"),
        #                                     ("Vaccination Rate", "@vaccination_rate")])

        hover = HoverTool(tooltips=[("Vaccination Rate", "@vaccination_rate")])

        p.add_tools(hover)

        # Plot using the color column from the source
        p.scatter('x', 'y', source=source, color='color', size=9, alpha=0.4)

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


def plot_comparison(country_code,
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
    Plots a bar chart comparing the normalized cluster and non-cluster statistics,
    with survey metrics means and correlations displayed on the plot. Beneath the bar chart,
    it also plots a distribution of vaccination rates within the cluster.

    Arguments:
    - cluster_stats: dict of statistics for the cluster
    - non_cluster_stats: dict of statistics for the non-cluster
    - cluster_label: int, the label of the cluster
    - cluster_count: int, the number of points in the cluster
    - non_cluster_count: int, the number of points outside the cluster
    - unnormalized_cluster_means: dict of unnormalized means for the cluster
    - unnormalized_non_cluster_means: dict of unnormalized means for non-cluster
    - survey_means_cluster: list, mean values for the cluster for survey metrics
    - correlations: dict of correlations between geospatial data types and survey metrics
    - cluster_vaccination_rates: list of vaccination rates for the cluster
    - non_cluster_vaccination_rates: list of vaccination rates for non-cluster
    - cluster_color: color for the cluster
    - save_to_disk: bool, whether to save the plot to disk (default is False)
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
    axs[1].set_title(f'{country_code}: Vaccination Distribution')
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



def wgs84_to_mercator(lon, lat):
    k = 6378137
    x = lon * (k * np.pi/180.0)
    y = np.log(np.tan((90 + lat) * np.pi/360.0)) * k
    return x, y


def create_geospatial_plot(source,
                           tooltips,
                           color_spec,
                           out_dir=None,
                           case=None,
                           plot_title=None,
                           color_bar=False,
                           color_bar_title=None,
                           symbol_size=10,
                           plot_width=950,
                           save_to_disk=True):
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
        p.scatter('mercator_x', 'mercator_y', source=source, fill_alpha=0.4, size=symbol_size,
                  color={'field': field, 'transform': color_mapper})

    elif hasattr(color_spec, 'transform') and isinstance(color_spec.transform, ColorMapper):
        color_mapper = color_spec.transform
        field = color_spec.field
        p.scatter('mercator_x', 'mercator_y', source=source, fill_alpha=0.4, size=symbol_size,
                  color={'field': field, 'transform': color_mapper})

    elif isinstance(color_spec, ColorMapper):
        color_mapper = color_spec
        p.scatter('mercator_x', 'mercator_y', source=source, fill_alpha=0.4, size=symbol_size,
                  color={'field': 'pc_value', 'transform': color_spec})  # Ensure 'pc_value' is valid

    else:
        p.scatter('mercator_x', 'mercator_y', source=source, fill_alpha=0.4, size=symbol_size, color="navy")
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