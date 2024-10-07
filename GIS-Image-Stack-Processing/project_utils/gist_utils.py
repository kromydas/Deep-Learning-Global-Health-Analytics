import os
import shutil
import numpy as np
import subprocess
import geopandas as gpd
import pandas as pd

import pyproj
from pyproj import Transformer
from pyproj import Proj
from pyproj import CRS

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from shapely.geometry import Point, Polygon

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from glob import glob

def run_gdalinfo(tif_path):

    # Construct the command string
    gdalinfo_command = ["gdalinfo", "-mm", tif_path]

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdalinfo_command, check=True, capture_output=True, text=True)
        print(result.stdout)

    except subprocess.CalledProcessError as e:

        print(f"An error occurred while running gdalinfo: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def gdal_crop(input_tif, output_tif, ulx, uly, lrx, lry, debug=False):

    if debug:
        # Debugging prints to check the file paths
        print(f"Input  TIF: {input_tif}")
        print(f"Output TIF: {output_tif}\n")

    # Run gdal_translate with the computed coordinates and file paths
    gdal_translate_command = [
        "gdal_translate",
        "-projwin", str(ulx), str(uly), str(lrx), str(lry),
        "-co", "COMPRESS=DEFLATE",
        "-co", "ZLEVEL=9",
        input_tif,
        output_tif
    ]

    if debug:
        print(f"gdal_translate command: {' '.join(gdal_translate_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdal_translate_command, check=True, capture_output=True, text=True)
        print(result.stdout)

    except subprocess.CalledProcessError as e:

        print(f"An error occurred while cropping data: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def gdal_replace_value(input_tif, output_tif, src_value, dst_value, debug=False):

    gdal_calc_command = [
        "gdal_calc.py",
        "--calc", f"numpy.where(A=={src_value},{dst_value},A)",
        "-A", input_tif,
        "--outfile", output_tif,
        "--overwrite",
        "--NoDataValue=None",
        "--co", "COMPRESS=DEFLATE",
        "--co", "ZLEVEL=9"
    ]

    if debug:
        print(f"gdal_calc.py command: {' '.join(gdal_calc_command)}\n")

    try:
        result = subprocess.run(gdal_calc_command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while replacing values: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def gdal_unset_nodata(input_tif, output_tif, debug=False):

    """Unset NoData value in the input TIFF file."""
    gdal_warp_command = [
        "gdalwarp",
        "-overwrite",
        "-co", "COMPRESS=DEFLATE",
        "-co", "ZLEVEL=9",
        input_tif,
        output_tif,
        "-dstnodata", "None"
    ]

    if debug:
        print(f"gdalwarp command: {' '.join(gdal_warp_command)}\n")

    try:
        result = subprocess.run(gdal_warp_command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while unsetting NoData values: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def gdal_set_nodata(input_tif, output_tif, src_nodata, dst_nodata, debug=False):

    if debug:
        print(f"Input TIF: {input_tif}")
        print(f"Final Output TIF: {output_tif}\n")

    # Check if source and destination NoData values are the same
    if src_nodata is not None and src_nodata == dst_nodata:
        # If NoData values are the same, simply copy the file
        shutil.copy(input_tif, output_tif)
        if debug:
            print(f"No change in NoData value needed. File copied to {output_tif}")
    else:
        # Construct the gdalwarp command to replace NoData values
        gdal_warp_command = [
            "gdalwarp",
            "-overwrite",
            "-co", "COMPRESS=DEFLATE",
            "-co", "ZLEVEL=9",
            input_tif,
            output_tif
        ]

        if src_nodata is not None:
            gdal_warp_command.extend(["-srcnodata", str(src_nodata)])

        gdal_warp_command.extend(["-dstnodata", str(dst_nodata)])

        if debug:
            print(f"gdalwarp command: {' '.join(gdal_warp_command)}\n")

        try:
            # Run the command, capture output and errors, ensure it handles errors via check=True
            result = subprocess.run(gdal_warp_command, check=True, capture_output=True, text=True)
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while replacing NoData values: {e}")
            if e.stderr:
                print("Error output:")
                print(e.stderr)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        if debug:
            print(f"Output TIF has been written to: {output_tif}")

def gdal_resample(input_tif, output_tif, resample_alg, x_res, y_res, debug=False):

    if debug:
        # Debugging prints to check the file paths
        print(f"Input  TIF: {input_tif}")
        print(f"Output TIF: {output_tif}\n")

    gdal_warp_command = [
        "gdalwarp",
        "-overwrite",
        "-tr", str(x_res), str(y_res),
        "-r", resample_alg,
        "-co", "COMPRESS=DEFLATE",
        "-co", "ZLEVEL=9",
        input_tif,
        output_tif
    ]

    if debug:
        print(f"gdalwarp command: {' '.join(gdal_warp_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdal_warp_command, check=True, capture_output=True, text=True)
        print(result.stdout)

    except subprocess.CalledProcessError as e:

        print(f"An error occurred while re-sampling data: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def sum_rasters(input_files, output_file, nodata_value=-9999, debug=False):
    """
    Sum a list of raster files using gdal_calc.py and save the result to a specified output file.
    NoData values are explicitly handled to prevent them from affecting the sum.

    Args:
        input_files (list of str): List of paths to the input raster files.
        output_file (str): The path to the output file where the result will be saved.
        nodata_value (float or int): The NoData value to exclude from calculations.
    """
    # Construct the command for gdal_calc.py
    gdal_calc_command = ["gdal_calc.py"]

    # Generate flags (-A, -B, etc.) and add files to the command
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    calc_expr = []

    for i, file_path in enumerate(input_files):
        if i < len(alphabet):  # Ensure there are not more files than letters in the alphabet
            letter = alphabet[i]
            gdal_calc_command.extend(['-' + letter, file_path])
            # Exclude NoData values from the sum
            calc_expr.append(f"({letter}!={nodata_value})*{letter}")
        else:
            print("Warning: More than 26 files detected; only the first 26 will be processed.")
            break

    # Adding the overwrite option and setting the NoData value for the output
    gdal_calc_command.extend([
        '--outfile', output_file,
        '--calc', '+'.join(calc_expr),
        '--NoDataValue', str(nodata_value),
        '--overwrite'
    ])

    if debug:
        print(f"gdal_calc.py command: {' '.join(gdal_calc_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdal_calc_command, check=True, capture_output=True, text=True)
        print(f"Sum of rasters successfully saved to: {output_file}")
        print("Output from command:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while summing the rasters: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def average_raster(input_file, output_file, divisor=365, nodata_value=-9999, debug=False):
    """
    Average a raster file by dividing each pixel by a specified divisor using gdal_calc.py.
    This version excludes NoData values from the calculation.

    Args:
        input_file (str): Path to the input raster file.
        output_file (str): Path to the output file where the result will be saved.
        divisor (int or float): The value to divide each pixel by.
        nodata_value (int or float): The NoData value to exclude from calculations.
    """

    # Check if the output file exists and delete it if it does
    if os.path.exists(output_file):
        if debug:
            print(f"Output file {output_file} exists and will be overwritten.")
        os.remove(output_file)

    # Construct the command for gdal_calc.py
    gdal_calc_command = [
        "gdal_calc.py",
        '-A', input_file,
        '--outfile', output_file,
        '--calc', f"(A!={nodata_value})*A/{divisor}",
        '--NoDataValue', str(nodata_value),
        '--overwrite'
    ]

    if debug:
        print(f"gdal_calc.py command: {' '.join(gdal_calc_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdal_calc_command, check=True, capture_output=True, text=True)
        print(f"Average operation successfully saved to: {output_file}")
        print("Output from command:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while averaging the rasters: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def average_rasters(input_files, output_file, nodata_value=-9999, debug=False):
    """
    Average multiple raster files by computing the average of each pixel.
    This version excludes NoData values from the calculation.

    Args:
        input_files (list of str): Paths to the input raster files.
        output_file (str): Path to the output file where the result will be saved.
        nodata_value (int or float): The NoData value to exclude from calculations.
    """

    # Check if the output file exists and delete it if it does
    if os.path.exists(output_file):
        if debug:
            print(f"Output file {output_file} exists and will be overwritten.")
        os.remove(output_file)

    # Prepare the input files for the gdal_calc.py command
    input_string = " + ".join([f"({chr(65 + i)}!={nodata_value})*{chr(65 + i)}" for i in range(len(input_files))])
    divisor = len(input_files)
    calc_string = f"({input_string})/{divisor}"

    # Construct the command for gdal_calc.py
    gdal_calc_command = [
        "gdal_calc.py",
        '--overwrite',
        '--NoDataValue', str(nodata_value),
        '--outfile', output_file
    ]

    # Add input files to the command
    for i, file in enumerate(input_files):
        gdal_calc_command.extend([f'-{chr(65 + i)}', file])

    # Add the calculation string to the command
    gdal_calc_command.extend(['--calc', calc_string])

    if debug:
        print(f"gdal_calc.py command: {' '.join(gdal_calc_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdal_calc_command, check=True, capture_output=True, text=True)
        print(f"Average operation successfully saved to: {output_file}")
        if debug:
            print("Output from command:")
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while averaging the rasters: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# def transform_to_mollweide(input_tif, output_tif, debug=False):
#     """
#     Transforms the coordinate reference system (CRS) of a GeoTIFF file to Mollweide projection using rasterio.
#
#     Parameters:
#     - input_tif (str): The file path to the input GeoTIFF that needs to be transformed.
#     - output_tif (str): The file path where the transformed GeoTIFF will be saved. If a file already
#       exists at this location, it will be overwritten.
#     - debug (bool, optional): If set to True, prints debugging information during the process.
#
#     Returns:
#     None. Outputs a transformed GeoTIFF file at the specified output path.
#     """
#     if debug:
#         print(f"Input TIF: {input_tif}")
#         print(f"Output TIF: {output_tif}")
#
#     # Check if the output file exists and delete it if it does
#     if os.path.exists(output_tif):
#         if debug:
#             print(f"Output file {output_tif} exists and will be overwritten.")
#         os.remove(output_tif)
#
#     # Mollweide PROJ string
#     mollweide_proj = '+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +axis=neu'
#
#     with rasterio.open(input_tif) as src:
#         transform, width, height = calculate_default_transform(
#             src.crs, mollweide_proj, src.width, src.height, *src.bounds)
#         kwargs = src.meta.copy()
#         kwargs.update({
#             'crs': mollweide_proj,
#             'transform': transform,
#             'width': width,
#             'height': height
#         })
#
#         with rasterio.open(output_tif, 'w', **kwargs) as dst:
#             for i in range(1, src.count + 1):
#                 reproject(
#                     source=rasterio.band(src, i),
#                     destination=rasterio.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src.crs,
#                     dst_transform=transform,
#                     dst_crs=mollweide_proj,
#                     resampling=Resampling.nearest)
#
#     if debug:
#         print("Transformation complete.")

def transform_to_CRS(input_tif, output_tif, proj_string, debug=False):
    """
    Transforms the coordinate reference system (CRS) of a GeoTIFF file to the projection specified by proj_string.
    Parameters:
    - input_tif (str): The file path to the input GeoTIFF that needs to be transformed.
    - output_tif (str): The file path where the transformed GeoTIFF will be saved. If a file already
      exists at this location, it will be overwritten.
    - debug (bool, optional): If set to True, prints debugging information during the process.

    Returns:
    None. Outputs a transformed GeoTIFF file at the specified output path.
    """
    if debug:
        print(f"Input TIF: {input_tif}")
        print(f"Output TIF: {output_tif}")

    # Check if the output file exists and delete it if it does
    if os.path.exists(output_tif):
        if debug:
            print(f"Output file {output_tif} exists and will be overwritten.")
        os.remove(output_tif)

    with rasterio.open(input_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, proj_string, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': proj_string,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=proj_string,
                    resampling=Resampling.nearest)

    if debug:
        print("Transformation complete.")

def transform_folder_to_CRS(input_folder, output_folder, proj_string, debug=False):
    """
    Transforms the coordinate reference system (CRS) of all single-channel GeoTIFF files in a folder
    to the projection specified by proj_string.

    Parameters:
    - input_folder (str): Path to the folder containing input GeoTIFF files to be transformed.
    - output_folder (str): Path to the folder where the transformed GeoTIFF files will be saved.
    - proj_string (str): The projection string (e.g., EPSG code) for the desired output CRS.
    - debug (bool, optional): If set to True, prints debugging information during the process.

    Returns:
    None. Outputs transformed GeoTIFF files in the specified output folder.
    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif') and 'Fmask' not in filename:
            input_tif = os.path.join(input_folder, filename)
            output_tif = os.path.join(output_folder, filename)

            if debug:
                print(f"Reprojecting {filename}...")

            # Reproject the GeoTIFF to the specified CRS
            transform_to_CRS(input_tif, output_tif, proj_string, debug)

    if debug:
        print("All files have been reprojected.")


def extract_cluster_data(shapefile_path, cluster_field, lat_field, lon_field, tolerance=1e-1):
    """
    Extracts cluster IDs and their corresponding GPS coordinates from a shapefile.

    Parameters:
        shapefile_path (str): The path to the shapefile.
        cluster_field (str): The field name for cluster IDs in the shapefile.
        lat_field (str): The field name for latitude values.
        lon_field (str): The field name for longitude values.
        tolerance (float): The tolerance for detecting coordinates close to (0, 0). Default is 1e-6.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with the selected columns.
            - list: A list of erroneous cluster IDs with coordinates near (0, 0).
    """
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Check if the specified fields exist in the dataframe
    if cluster_field not in gdf.columns or lat_field not in gdf.columns or lon_field not in gdf.columns:
        raise ValueError("One or more field names do not exist in the shapefile.")

    # Extract the necessary information
    cluster_data = gdf[[cluster_field, lat_field, lon_field]]

    # Convert cluster IDs to integers
    cluster_data[cluster_field] = cluster_data[cluster_field].astype(float).astype(int)

    # Detect erroneous clusters with coordinates near (0, 0) using a tolerance
    erroneous_clusters = cluster_data[
        (cluster_data[lat_field].abs() < tolerance) &
        (cluster_data[lon_field].abs() < tolerance)
    ]

    # Extract the erroneous cluster IDs and coordinates
    erroneous_cluster_ids = erroneous_clusters[cluster_field].tolist()
    erroneous_cluster_coords = erroneous_clusters[[lat_field, lon_field]].values.tolist()

    if erroneous_cluster_ids:
        # Print the cluster IDs along with their lat/lon coordinates
        print(f"Erroneous clusters detected:")
        for cid, (lat, lon) in zip(erroneous_cluster_ids, erroneous_cluster_coords):
            print(f"Cluster ID: {cid}, Latitude: {lat}, Longitude: {lon}")

        # Remove erroneous clusters from the data
        cluster_data = cluster_data[
            (cluster_data[lat_field].abs() >= tolerance) |
            (cluster_data[lon_field].abs() >= tolerance)
        ]

    return cluster_data, erroneous_cluster_ids

def convert_cluster_coordinates(cluster_data, src_crs, dst_crs):
    """
    Converts a list of cluster coordinates from a source CRS to a destination CRS using either EPSG codes or PROJ strings.

    Each cluster's latitude and longitude are transformed to the specified destination CRS. The
    function returns a list of tuples, each containing the cluster ID and its transformed coordinates.

    Parameters:
    - cluster_data (list of tuples): A list of tuples, where each tuple contains (cluster_id, lat, lon).
    - src_crs (str): The EPSG code or PROJ string of the source coordinate reference system.
    - dst_crs (str): The EPSG code or PROJ string of the destination coordinate reference system.

    Returns:
    list of tuples: A list where each tuple contains (cluster_id, x, y), with 'x' and 'y' being the
    coordinates in the destination CRS.

    Example:
    >>> cluster_data = [(1, 34.05, -118.25), (2, 40.7128, -74.0060)]
    >>> convert_cluster_coordinates(cluster_data, 'EPSG:4326', '+proj=moll +lon_0=0 +datum=WGS84')
    [(1, coordinates...), (2, coordinates...)]
    """
    # Initialize projection objects
    src_proj = Proj(src_crs)
    dst_proj = Proj(dst_crs)

    # Create a transformer
    transformer = Transformer.from_proj(src_proj, dst_proj)

    transformed_data = []
    for cluster_id, lat, lon in cluster_data:
        # Transform coordinates
        # x, y = transform(src_proj, dst_proj, lon, lat)
        # Transforming the point to projected coordinates using the transformer
        x, y = transformer.transform(lat, lon)
        transformed_data.append((int(cluster_id), x, y))
    return transformed_data


def load_raster(file_path, expected_crs, expected_pixel_size=None):
    
    src = rasterio.open(file_path)  # Open the raster file
    raster_crs = src.crs

    # Create CRS objects for comparison
    expected_crs_obj = CRS(expected_crs)
    raster_crs_obj = CRS(raster_crs)

    # Compare the CRS objects for equivalency
    crs_match = raster_crs_obj == expected_crs_obj

    pixel_size_match = True  # Default to True if no pixel size is expected
    if expected_pixel_size:
        pixel_width, pixel_height = src.transform[0], abs(src.transform[4])
        pixel_size_match = (pixel_width, pixel_height) == expected_pixel_size

    return src, crs_match, pixel_size_match


def find_points_within_raster(raster, points, target_crs, debug=False):
    """
    Filters a list of points, returning those within the raster's boundaries.
    All points must be tuples in the format (cluster_id, x, y).

    Parameters:
        raster (rasterio.io.DatasetReader): The raster object which provides the bounds.
        points (list): A list of points in the format (cluster_id, x, y), where x and y are coordinates.
        target_crs (str): The target coordinate reference system to which points and raster bounds are transformed.
        debug (bool): If True, prints debug information about the transformation and filtering process.

    Returns:
        list: A list of dictionaries, each containing a 'point' (Shapely Point object) and 'cluster_id',
              for points that are within the raster bounds.
    """
    target_crs = pyproj.CRS(target_crs)  # Define the target CRS from the given string

    # Get the raster bounds and transform them to EPSG:3035
    left, bottom, right, top = raster.bounds
    source_crs = raster.crs

    try:
        transformed_bounds = transform_bounds(source_crs, target_crs, left, bottom, right, top)
    except Exception as e:
        print(f"Error transforming bounds: {e}")
        return []

    ul_x, ul_y, lr_x, lr_y = transformed_bounds
    ur_x, ur_y = lr_x, ul_y
    ll_x, ll_y = ul_x, lr_y

    ul = (ul_x, ul_y)
    ur = (ur_x, ur_y)
    lr = (lr_x, lr_y)
    ll = (ll_x, ll_y)

    aoi_bounds = Polygon([ul, ur, lr, ll])

    if debug:
        coordinates = list(aoi_bounds.exterior.coords)
        for x, y in coordinates:
            print(f"bbox: ({x:.3f}, {y:.3f})")

    # Define transformation from the source CRS of the points to EPSG:3035
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Filter points based on whether they lie within the raster bounds
    points_within_raster = []
    for cluster_id, x, y in points:
        try:
            if debug:
                print(f"Transforming point ({x}, {y}) with cluster_id {cluster_id}")
            x_transformed, y_transformed = transformer.transform(x, y)
            if not (float('inf') in (x_transformed, y_transformed) or float('-inf') in (x_transformed, y_transformed)):
                point = Point(x_transformed, y_transformed)
                if debug:
                    print(f"Checking point {point} with cluster_id {cluster_id}")
                if aoi_bounds.contains(point):
                    points_within_raster.append({'point': point, 'cluster_id': cluster_id})
                    if debug:
                        print(f"Point {point} is within the raster bounds.")
                else:
                    if debug:
                        print(f"Point {point} is NOT within the raster bounds.\n")
            else:
                if debug:
                    print(f"Invalid coordinates for point: {x}, {y} -> Transformed: ({x_transformed}, {y_transformed})")
        except Exception as e:
            if debug:
                print(f"Error transforming point ({x}, {y}): {e}")

    return points_within_raster


def find_nearest_vertex_rasterio(raster, x, y):
    """
    Convert x,y coordinates to the nearest vertex of the pixel

    Parameters:
        raster (rasterio.io.DatasetReader): The rasterio dataset object.
        x (float): X coordinate
        y (float): Y coordinate

    Returns:
        tuple: The x,y coordinates of the nearest vertex of a pixel.
    """
    # Get the pixel coordinates of the center of the pixel that contains the point
    px, py = raster.index(x, y)

    # Get the x,y coordinates for the center of this pixel
    center_x, center_y = raster.xy(px, py)

    # Determine the half pixel size
    half_res_x = raster.res[0] / 2
    half_res_y = raster.res[1] / 2

    # Adjust to the nearest vertex by comparing input coordinates to the pixel center
    if x < center_x:
        vertex_x = center_x - half_res_x
    else:
        vertex_x = center_x + half_res_x

    if y < center_y:
        vertex_y = center_y - half_res_y
    else:
        vertex_y = center_y + half_res_y

    return (vertex_x, vertex_y)


def crop_raster_rasterio(rasterio_dataset, points, filename_prefix, filename_suffix, output_folder, tile_size=224, debug=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, point_data in enumerate(points, start=1):
        if isinstance(point_data, dict) and 'point' in point_data and 'cluster_id' in point_data:
            point = point_data['point']
            cluster_id = point_data['cluster_id']
        else:
            print("Invalid point data format. Expected a dictionary with 'point' and 'cluster_id'.")
            continue

        center_x, center_y = find_nearest_vertex_rasterio(rasterio_dataset, point.x, point.y)

        if debug:
           print(f"Converted ({point.x}, {point.y}) to Vertex ({center_x}, {center_y})")

        # Convert coordinates to pixel coordinates for cropping
        py, px = rasterio_dataset.index(center_x, center_y)
        px_ul_x = max(px - tile_size // 2, 0)  # Ensure not less than 0
        px_ul_y = max(py - tile_size // 2, 0)  # Ensure not less than 0

        px_lr_x = min(px_ul_x + tile_size, rasterio_dataset.width)  # Ensure does not exceed raster width
        px_lr_y = min(px_ul_y + tile_size, rasterio_dataset.height)  # Ensure does not exceed raster height

        if debug:
            print(f"Calculated pixel coords: Upper Left ({px_ul_x}, {px_ul_y}), Lower Right ({px_lr_x}, {px_lr_y})")

        # Define window ensuring it stays within raster bounds
        if (px_lr_x > px_ul_x) and (px_lr_y > px_ul_y):  # Check if the window is valid
            window = rasterio.windows.Window(px_ul_x, px_ul_y, px_lr_x - px_ul_x, px_lr_y - px_ul_y)
            transform = rasterio_dataset.window_transform(window)

            output_filename = f"{filename_prefix}_{idx}_C-{cluster_id}_{filename_suffix}.tif"
            output_raster_path = os.path.join(output_folder, output_filename)

            # Read the window and write it to a new file
            with rasterio.open(
                output_raster_path,
                'w',
                driver='GTiff',
                height=window.height,
                width=window.width,
                count=rasterio_dataset.count,
                dtype=rasterio_dataset.dtypes[0],
                crs=rasterio_dataset.crs,
                transform=transform
            ) as dst:
                dst.write(rasterio_dataset.read(window=window, boundless=True, fill_value=0))
        else:
            print(f"No valid crop window for point at index {idx} with cluster ID {cluster_id}. Skipping...")

    print(f"Crops are saved in {output_folder}")

def rasterio_replace_nodata(input_tif, output_tif, dst_nodata, src_nodata=None, debug=False):
    with rasterio.open(input_tif) as src:
        data = src.read(1)  # Read the first band (assuming a single band for simplicity)

        # Replace the source NoData values with the destination NoData value
        if src_nodata is not None and src_nodata != dst_nodata:
            mask = data == src_nodata
            data[mask] = dst_nodata

        profile = src.profile
        profile.update(nodata=dst_nodata, compress='deflate', zlevel=9)

        if debug:
            print(f"Writing to {output_tif} with NoData value set to {dst_nodata}")

        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(data, 1)


def load_and_plot_geotiffs(file_path, plot_data=True, cmap='gray', precision=np.float32, plot_geo_coords=False,
                           bad_value=-9999.9, clip_percentile=0, figsize=(12,6)):
    files = glob(file_path)
    files.sort()  # Ensure files are processed in a consistent order

    stats_df = pd.DataFrame(columns=["File", "Min", "Max", "Mean", "Median", "NoData Count", "Bad Value Count", "NoData Percent", "Bad Value Percent"])

    for file in files:
        base_name, extension = os.path.splitext(os.path.basename(file))

        with rasterio.open(file) as src:
            data = src.read(1).astype(precision)
            no_data = src.nodata
            print(f"NoData value for {file}: {no_data}")  # Print NoData value set in the file

            # Create masks for NoData and Bad Values
            mask_no_data = data == no_data if no_data is not None else np.zeros_like(data, dtype=bool)
            mask_bad_value = data <= bad_value  # Adjusted to catch any value less than or equal to bad_value

            # Filter out invalid data
            valid_data = data[~mask_no_data & ~mask_bad_value]

            # Calculate statistics
            no_data_count = np.sum(mask_no_data)
            bad_value_count = np.sum(mask_bad_value)
            total_pixels = data.size
            no_data_percent = (no_data_count / total_pixels) * 100
            bad_value_percent = (bad_value_count / total_pixels) * 100

            if valid_data.size > 0 and np.isfinite(valid_data).any():
                if clip_percentile > 0:
                    min_val = np.nanpercentile(valid_data, clip_percentile)
                    max_val = np.nanpercentile(valid_data, 100 - clip_percentile)
                else:
                    min_val = np.nanmin(valid_data)
                    max_val = np.nanmax(valid_data)

                mean_val = np.nanmean(valid_data)
                median_val = np.nanmedian(valid_data)
            else:
                # Handle cases where valid_data is empty or all-NA
                min_val = max_val = mean_val = median_val = 'No valid data'

            # Prepare a new row for the DataFrame
            new_row = {
                "File": base_name[0:10],
                "Min": min_val,
                "Max": max_val,
                "Mean": mean_val,
                "Median": median_val,
                "NoData Count": no_data_count,
                "Bad Value Count": bad_value_count,
                "NoData Percent": no_data_percent,
                "Bad Value Percent": bad_value_percent
            }

            # Only concatenate if the new row is not empty or all-NA
            if any(pd.notna(val) for val in new_row.values()):
                stats_df = pd.concat([stats_df, pd.DataFrame([new_row])], ignore_index=True)

            if plot_data:
                # Normalize data
                low_end = 0
                high_end = 1
                norm_data = np.clip((data - min_val) / (max_val - min_val), low_end, high_end)
                norm_data[mask_no_data | mask_bad_value] = -1  # Set both NoData and bad values to -1 for visualization

                plt.figure(figsize=figsize)

                # Modify the colormap to add magenta for values set to -1
                new_cmap = plt.get_cmap(cmap)
                new_cmap.set_under('magenta')  # Set color for out-of-range low values (below 0)

                extent = src.bounds if plot_geo_coords else None

                plt.imshow(norm_data, cmap=new_cmap, norm=Normalize(vmin=low_end, vmax=high_end), extent=extent, aspect='equal')
                plt.xlabel('Pixel X coordinate' if not plot_geo_coords else 'Longitude')
                plt.ylabel('Pixel Y coordinate' if not plot_geo_coords else 'Latitude')

                plt.show()

    return stats_df












