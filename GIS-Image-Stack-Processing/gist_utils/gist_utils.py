import os
import shutil
import subprocess
import geopandas as gpd
from osgeo import gdal, ogr, osr
import pyproj
from pyproj import Transformer
from pyproj import Proj, transform
from pyproj import CRS
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
from shapely.geometry import Point, Polygon

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
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

def get_epsg_code(utm_zone):
    """
    Converts a UTM zone string into an EPSG code string.

    Parameters:
        utm_zone (str): The UTM zone string, which includes the zone number followed by 'N' for the Northern Hemisphere
                        or 'S' for the Southern Hemisphere (e.g., '43N', '36S').

    Returns:
        str: The corresponding EPSG code as a string (e.g., 'EPSG:32643' for '43N').

    Raises:
        ValueError: If the utm_zone does not end with 'N' or 'S'.

    Examples:
        >>> get_epsg_code('43N')
        'EPSG:32643'
        >>> get_epsg_code('36S')
        'EPSG:32736'
    """
    # Extract the UTM zone number and hemisphere
    zone_number = int(utm_zone[:-1])
    hemisphere = utm_zone[-1].upper()

    # Determine the EPSG code based on hemisphere
    if hemisphere == 'N':
        return f"EPSG:326{zone_number}"
    elif hemisphere == 'S':
        return f"EPSG:327{zone_number}"
    else:
        raise ValueError("UTM zone must end in 'N' or 'S'.")


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

def utm_zone_longitude_bounds(epsg_code):

    # Extract the UTM zone number and hemisphere from the input code
    # Extract the UTM zone number and hemisphere from the EPSG code
    print("epsg_code: ", epsg_code)
    if isinstance(epsg_code, str) and epsg_code.startswith("EPSG:"):
        zone_code = int(epsg_code.split(":")[1])
        zone_number = zone_code % 100  # Extract the last two digits for the zone number
        if zone_code < 32601 or (zone_code > 32660 and zone_code < 32701) or zone_code > 32760:
            raise ValueError("Invalid UTM zone EPSG code. It should be between 32601-32660 for the north and 32701-32760 for the south.")
    else:
        raise ValueError("Invalid EPSG code format. Please provide a string in the format 'EPSG:32642'.")

    # Calculate the west and east longitude boundaries for the UTM zone
    west_longitude = (zone_number - 1) * 6 - 180
    east_longitude = west_longitude + 6

    return west_longitude, east_longitude

def transform_to_utm(input_tif, output_tif, utm_zone, debug=False):
    """
    Transforms the coordinate reference system (CRS) of a GeoTIFF file to a specified UTM zone.

    This function utilizes `gdalwarp` from the GDAL library to reproject the input GeoTIFF to the
    Universal Transverse Mercator (UTM) coordinate system corresponding to the provided UTM zone.
    Parameters:
    - input_tif (str): The file path to the input GeoTIFF that needs to be transformed.
    - output_tif (str): The file path where the transformed GeoTIFF will be saved. If a file already
      exists at this location, it will be overwritten.
    - utm_zone (str): The UTM zone to which the input GeoTIFF will be transformed. It should be
      specified as a string, e.g., 'EPSG:32633' for UTM zone 33N.
    - debug (bool, optional): If set to True, prints debugging information during the process, including
      file paths and the full gdalwarp command. Default is False.

    Returns:
    None. Outputs a transformed GeoTIFF file at the specified output path.

    Raises:
    subprocess.CalledProcessError: If the gdalwarp command fails due to a processing error, detailed
      error information will be printed.
    Exception: Catches and prints unexpected errors during execution.

    Example:
    >>> transform_to_utm('path/to/input.tif', 'path/to/output.tif', 'EPSG:32633', debug=True)
    """
    if debug:
        # Debugging prints to check the file paths
        print(f"Input  TIF: {input_tif}")
        print(f"Output TIF: {output_tif}\n")

    # Check if the output file exists and delete it if it does
    if os.path.exists(output_tif):
        if debug:
            print(f"Output file {output_tif} exists and will be overwritten.")
        os.remove(output_tif)

    # Construct the gdalwarp command
    gdalwarp_command = [
        "gdalwarp",
        "-t_srs", utm_zone,  # Target SRS
        "-overwrite",
        "-co", "COMPRESS=DEFLATE",
        "-co", "ZLEVEL=9",
        input_tif,
        output_tif
    ]

    if debug:
        print(f"gdalwarp command: {' '.join(gdalwarp_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdalwarp_command, check=True, capture_output=True, text=True)
        print(result.stdout)

    except subprocess.CalledProcessError as e:

        print(f"An error occurred while summing the rasters: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def transform_to_mollweide(input_tif, output_tif, debug=False):
    """
    Transforms the coordinate reference system (CRS) of a GeoTIFF file to Mollweide projection using rasterio.

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

    # Mollweide PROJ string
    mollweide_proj = '+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +axis=neu'

    with rasterio.open(input_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, mollweide_proj, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': mollweide_proj,
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
                    dst_crs=mollweide_proj,
                    resampling=Resampling.nearest)

    if debug:
        print("Transformation complete.")

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

def latlon_to_utm(lat, lon, src_crs='EPSG:4326', dst_crs='EPSG:32642'):
    """
    Transforms latitude and longitude coordinates to UTM.

    Parameters:
    - lat (float): Latitude of the coordinate to transform.
    - lon (float): Longitude of the coordinate to transform.
    - src_crs (str): Source CRS, defaults to WGS 84.
    - dst_crs (str): Destination UTM CRS, defaults to UTM Zone 42N.

    Returns:
    (float, float): The transformed coordinates (x, y) in UTM.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def utm_to_latlon(x, y, src_crs='EPSG:32642', dst_crs='EPSG:4326'):
    """
    Transforms UTM coordinates to latitude and longitude.

    Parameters:
    - x (float): Easting of the coordinate to transform.
    - y (float): Northing of the coordinate to transform.
    - src_crs (str): Source UTM CRS, defaults to UTM Zone 42N.
    - dst_crs (str): Destination CRS, defaults to WGS 84.

    Returns:
    (float, float): The transformed coordinates (lat, lon).
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon

def extract_cluster_info(shapefile_path, cluster_field, lat_field, lon_field):
    """
    Extracts cluster IDs and their corresponding GPS coordinates from a shapefile.

    Parameters:
        shapefile_path (str): The path to the shapefile.
        cluster_field (str): The field name for cluster IDs in the shapefile.
        lat_field (str): The field name for latitude values.
        lon_field (str): The field name for longitude values.

    Returns:
        list of tuples: A list where each tuple contains (cluster_id, latitude, longitude).
    """
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Check if the specified fields exist in the dataframe
    if cluster_field not in gdf.columns or lat_field not in gdf.columns or lon_field not in gdf.columns:
        raise ValueError("One or more field names do not exist in the shapefile.")

    # Extract the necessary information
    data = gdf[[cluster_field, lat_field, lon_field]]

    # Convert the dataframe to a list of tuples
    cluster_info = [tuple(x) for x in data.to_numpy()]

    return cluster_info

def convert_cluster_coordinates_EPSG(cluster_data, src_crs, dst_crs):
    """
    Converts a list of cluster coordinates from a source CRS to a destination CRS.

    Each cluster's latitude and longitude are transformed to the specified destination CRS. The
    function returns a list of tuples, each containing the cluster ID and its transformed coordinates.

    Parameters:
    - cluster_data (list of tuples): A list of tuples, where each tuple contains (cluster_id, lat, lon).
    - src_crs (str): The EPSG code of the source coordinate reference system.
    - dst_crs (str): The EPSG code of the destination coordinate reference system.

    Returns:
    list of tuples: A list where each tuple contains (cluster_id, x, y), with 'x' and 'y' being the
    coordinates in the destination CRS.

    Example:
    >>> cluster_data = [(1, 34.05, -118.25), (2, 40.7128, -74.0060)]
    >>> convert_cluster_coordinates(cluster_data, 'EPSG:4326', 'EPSG:32611')
    [(1, 500000.0, 3762155.0), (2, 500000.0, 4512337.0)]
    """
    transformed_data = []
    for cluster_id, lat, lon in cluster_data:
        x, y = latlon_to_utm(lat, lon, src_crs, dst_crs)
        transformed_data.append((int(cluster_id), x, y))
    return transformed_data

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


def load_raster_EPSG(file_path, expected_crs, expected_pixel_size=None):
    """
    Loads a GeoTIFF file, confirms its CRS, and optionally checks the pixel size.

    Parameters:
        file_path (str): Path to the GeoTIFF file.
        expected_crs (str): The expected CRS in EPSG code format.
        expected_pixel_size (tuple, optional): A tuple (pixel_width, pixel_height) representing the expected pixel size in meters.

    Returns:
        rasterio DatasetReader object, bool, bool: The rasterio object, a boolean indicating if the CRS matches,
        and a boolean indicating if the pixel size matches (if expected_pixel_size is provided).
    """
    src = rasterio.open(file_path)  # Open the raster file
    raster_crs = src.crs

    # Check if the raster CRS matches the expected CRS
    crs_match = raster_crs.to_string() == expected_crs

    pixel_size_match = True  # Default to True if no pixel size is expected
    if expected_pixel_size:
        # Extract pixel dimensions from the geotransform (width is src.transform[0], height is abs(src.transform[4]) due to north-up convention)
        pixel_width, pixel_height = src.transform[0], abs(src.transform[4])
        # Compare as tuples
        pixel_size_match = (pixel_width, pixel_height) == expected_pixel_size

    return src, crs_match, pixel_size_match

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

def extract_latitudes_from_raster_utm(raster):

    # Define the transformer to convert UTM to WGS 84 (lat, lon)
    transformer = Transformer.from_crs(raster.crs, 'EPSG:4326', always_xy=True)

    # Get the bounds of the raster
    bounds = raster.bounds

    # Transform the coordinates
    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)

    return max_lat, min_lat

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

def transform_raster_points(raster, points, dst_crs, debug=False):
    """
    Transforms a list of point coordinates from the raster's source CRS to EPSG:3035.

    Parameters:
        raster (rasterio.io.DatasetReader): The raster object which provides the bounds.
        points (list): A list of points in the format (cluster_id, x, y), where x and y are coordinates.

    Returns:
        list: A list of dictionaries, each containing a 'point' (Shapely Point object) and 'cluster_id',
              with coordinates transformed to EPSG:3035.
    """
    # Define the target CRS (EPSG:3035)
    target_crs = pyproj.CRS(dst_crs)
    source_crs = raster.crs

    # Define transformation from the source CRS to target CRS
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Transform and collect points
    transformed_points = []
    for cluster_id, x, y in points:
        try:
            if debug:
                print(f"Transforming point ({x}, {y}) with cluster_id {cluster_id}")
            x_transformed, y_transformed = transformer.transform(x, y)
            if not (float('inf') in (x_transformed, y_transformed) or float('-inf') in (x_transformed, y_transformed)):
                point = Point(x_transformed, y_transformed)
                transformed_points.append({'point': point, 'cluster_id': cluster_id})
                if debug:
                    print(f"Transformed point ({x_transformed}, {y_transformed}) with cluster_id {cluster_id}")
            else:
                if debug:
                    print(f"Invalid coordinates for point: {x}, {y} -> Transformed: ({x_transformed}, {y_transformed})")
        except Exception as e:
            if debug:
                print(f"Error transforming point ({x}, {y}): {e}")

    return transformed_points


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

def find_nearest_vertex_gdal(gdal_raster, x, y):
    """
    Convert UTM coordinates to the nearest vertex of the pixel using GDAL.

    Parameters:
        gdal_raster (gdal.Dataset): The raster dataset opened with GDAL.
        x (float): X coordinate in UTM.
        y (float): Y coordinate in UTM.

    Returns:
        tuple: The UTM coordinates of the nearest vertex of a pixel.
    """
    # Get the geotransform from the GDAL raster
    gt = gdal_raster.GetGeoTransform()

    # Convert from UTM coordinates to pixel coordinates
    px = int((x - gt[0]) / gt[1])  # How many pixels away from the origin (x)
    py = int((y - gt[3]) / gt[5])  # How many pixels away from the origin (y)

    # Calculate the UTM coordinates for the nearest vertex
    # Nearest vertex coordinates (adjusting to the pixel's corner rather than center)
    vertex_x = gt[0] + px * gt[1]
    vertex_y = gt[3] + py * gt[5]

    return (vertex_x, vertex_y)

def crop_raster_gdal(rasterio_dataset, points_with_metadata, filename_prefix, filename_suffix, output_folder,
                     tile_size=224, debug=False):
    """Crop raster based on provided UTM coordinates, saving tiles to the specified folder using GDAL."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        if debug:
            print(f"Created output folder at {output_folder}")

    # Extract the file path from the rasterio dataset
    if isinstance(rasterio_dataset, rasterio.io.DatasetReader):
        raster_path = rasterio_dataset.name
        if debug:
            print(f"Using raster from path: {raster_path}")
    else:
        raise ValueError("The input must be a rasterio dataset.")

    # Open the raster file using GDAL
    gdal_raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if gdal_raster is None:
        raise FileNotFoundError("Raster file could not be opened.")
    else:
        if debug:
            print(f"Raster file {raster_path} opened successfully.")

    for idx, point_data in enumerate(points_with_metadata, start=1):
        if isinstance(point_data, dict) and 'point' in point_data and 'cluster_id' in point_data:
            point = point_data['point']
            cluster_id = point_data['cluster_id']
            if debug:
               print(f"Processing point {idx}: {point}, cluster ID: {cluster_id}")
        else:
            print("Invalid point data format. Expected a dict with 'point' and 'cluster_id'.")
            continue

        # Assume find_nearest_vertex function is defined elsewhere
        center_utm_x, center_utm_y = find_nearest_vertex_gdal(gdal_raster, point.x, point.y)
        if debug:
            print(f"Nearest vertex UTM coordinates: ({center_utm_x}, {center_utm_y})")

        # Convert UTM coordinates back to pixel coordinates for cropping
        gt = gdal_raster.GetGeoTransform()
        px = int((center_utm_x - gt[0]) / gt[1])  # x pixel
        py = int((center_utm_y - gt[3]) / gt[5])  # y pixel

        px_ul_x = px - tile_size // 2
        px_ul_y = py - tile_size // 2

        # Define the window for cropping
        window_width = min(tile_size, gdal_raster.RasterXSize - px_ul_x)
        window_height = min(tile_size, gdal_raster.RasterYSize - px_ul_y)

        if debug:
            print(f"Pixel coordinates: (x: {px}, y: {py})")
            print(f"Upper left pixel coordinates for crop: (x: {px_ul_x}, y: {px_ul_y})")
            print(f"Window dimensions: {window_width}x{window_height}")

        # Read the data from the window
        data = gdal_raster.ReadRaster(px_ul_x, px_ul_y, window_width, window_height)

        # Create the output file
        output_filename = f"{filename_prefix}_{idx:03d}_C-{cluster_id:03d}_{filename_suffix}.tif"
        output_raster_path = os.path.join(output_folder, output_filename)
        if debug:
            print(f"Window dimensions: {window_width}x{window_height}")

        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(output_raster_path, window_width, window_height, 1,
                                   gdal_raster.GetRasterBand(1).DataType)
        out_raster.SetGeoTransform([gt[0] + px_ul_x * gt[1], gt[1], 0, gt[3] + px_ul_y * gt[5], 0, gt[5]])
        out_raster.SetProjection(gdal_raster.GetProjection())
        out_raster.GetRasterBand(1).WriteRaster(0, 0, window_width, window_height, data)
        out_raster.FlushCache()

    print(f"Crops are saved in {output_folder}")

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

            output_filename = f"{filename_prefix}_{idx:03d}_C-{cluster_id:03d}_{filename_suffix}.tif"
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


def rasterio_copy_or_replace_nodata(input_tif, output_tif, dst_nodata, src_nodata=None, debug=False):
    if src_nodata == dst_nodata:
        # If source and destination NoData values are the same, copy the file directly.
        shutil.copy(input_tif, output_tif)
        if debug:
            print(f"Copied {input_tif} to {output_tif} without alteration.")
    else:
        # Process the raster data to replace the NoData value.
        with rasterio.open(input_tif) as src:
            data = src.read(1)  # Read the first band (assuming a single band for simplicity)

            # Replace the source NoData values with the destination NoData value
            if src_nodata is not None:
                mask = data == src_nodata
                data[mask] = dst_nodata

            profile = src.profile
            profile.update(nodata=dst_nodata, compress='deflate', zlevel=9)

            if debug:
                print(f"Writing to {output_tif} with NoData value set to {dst_nodata}")

            with rasterio.open(output_tif, 'w', **profile) as dst:
                dst.write(data, 1)


def load_and_plot_geotiffs(file_path, plot_data=True, cmap='gray', precision=np.float32, plot_geo_coords=False,
                           bad_value=-9999, clip_percentile=0):
    files = glob(file_path)
    files.sort()  # Sort the files to ensure ordered processing
    stats_df = pd.DataFrame(
        columns=["File", "Min", "Max", "Mean", "Median", "NoData Count", "Bad Value Count", "NoData Percent",
                 "Bad Value Percent"])

    for file in files:
        base_name, extension = os.path.splitext(os.path.basename(file))

        with rasterio.open(file) as src:
            # Read the first band of the raster data from the file
            data = src.read(1).astype(precision)
            # Retrieve the NoData value from the raster's metadata
            no_data = src.nodata
            print(f"NoData value for {file}: {no_data}")  # Print NoData value set in the file

            # Create 2D boolean masks
            mask_no_data = data == no_data  # Mask for NoData pixels
            mask_bad_value = data == bad_value  # Mask for bad value pixels

            # Create a 2D array of valid data where invalid values are set to np.nan
            valid_data = np.where((~mask_no_data) & (~mask_bad_value), data, np.nan)

            # Compute statistics from valid data, ignoring NaN values
            no_data_count = np.sum(mask_no_data)
            bad_value_count = np.sum(mask_bad_value)
            total_pixels = data.size
            no_data_percent = (no_data_count / total_pixels) * 100
            bad_value_percent = (bad_value_count / total_pixels) * 100

            if clip_percentile > 0:
                min_val = np.nanpercentile(valid_data, clip_percentile) if np.isfinite(
                    valid_data).any() else 'No valid data'
                max_val = np.nanpercentile(valid_data, 100 - clip_percentile) if np.isfinite(
                    valid_data).any() else 'No valid data'
            else:
                min_val = np.nanmin(valid_data) if np.isfinite(valid_data).any() else 'No valid data'
                max_val = np.nanmax(valid_data) if np.isfinite(valid_data).any() else 'No valid data'

            mean_val = np.nanmean(valid_data) if np.isfinite(valid_data).any() else 'No valid data'
            median_val = np.nanmedian(valid_data) if np.isfinite(valid_data).any() else 'No valid data'

            # Append the results to the DataFrame
            stats_df = pd.concat([stats_df, pd.DataFrame([{
                "File": base_name[0:10],
                "Min": min_val,
                "Max": max_val,
                "Mean": mean_val,
                "Median": median_val,
                "NoData Count": no_data_count,
                "Bad Value Count": bad_value_count,
                "NoData Percent": no_data_percent,
                "Bad Value Percent": bad_value_percent
            }])], ignore_index=True)

            if plot_data:
                # Normalize the data, ignoring NaN values
                norm_data = np.clip((valid_data - min_val) / (max_val - min_val), 0, 1)

                # Mark bad values distinctly
                norm_data[mask_bad_value] = -1

                plt.figure(figsize=(8, 6))
                colors = plt.get_cmap(cmap)(np.linspace(0, 1, 256))
                # Add magenta at the start for bad values (with .5 transparency)
                colors = np.vstack(([1, 0, 1, .5], colors))
                new_cmap = ListedColormap(colors)

                if plot_geo_coords:
                    plt.imshow(norm_data, cmap=new_cmap, norm=Normalize(vmin=-1, vmax=1), extent=src.bounds)
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                else:
                    plt.imshow(norm_data, cmap=new_cmap, norm=Normalize(vmin=-1, vmax=1))
                    plt.xlabel('Pixel X coordinate')
                    plt.ylabel('Pixel Y coordinate')

                plt.show()

    return stats_df








