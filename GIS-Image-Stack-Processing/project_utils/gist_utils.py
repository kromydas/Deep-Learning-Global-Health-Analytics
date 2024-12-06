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
    """
       Runs the GDAL gdalinfo command on a GeoTIFF file to gather metadata information, including minimum and maximum values.

       Parameters:
       -----------
       tif_path : str
           Path to the GeoTIFF file for which metadata information is to be gathered.

       Returns:
       --------
       None
           This function does not return anything. The metadata information is printed to the console.

       Raises:
       -------
       subprocess.CalledProcessError
           If the gdalinfo command fails, an error message is printed and this exception is raised.
       Exception
           If any other error occurs, it will print an error message describing the issue.

       Example:
       --------
       run_gdalinfo(tif_path='example_file.tif')
       """
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
    """
      Crops a GeoTIFF file to a specified bounding box using GDAL's gdal_translate command.

      Parameters:
      -----------
      input_tif : str
          Path to the input GeoTIFF file.
      output_tif : str
          Path to the output GeoTIFF file where the cropped data will be saved.
      ulx : float
          Upper left X coordinate of the cropping window.
      uly : float
          Upper left Y coordinate of the cropping window.
      lrx : float
          Lower right X coordinate of the cropping window.
      lry : float
          Lower right Y coordinate of the cropping window.
      debug : bool, optional (default=False)
          If True, prints debugging information, including the input and output paths,
          as well as the constructed GDAL command.

      Returns:
      --------
      None
          This function does not return anything. The result is saved as the output GeoTIFF file.

      Raises:
      -------
      subprocess.CalledProcessError
          If the gdal_translate command fails, an error message is printed and this exception is raised.
      Exception
          If any other error occurs, it will print an error message describing the issue.

      Example:
      --------
      gdal_crop(
          input_tif='input_file.tif',
          output_tif='output_file.tif',
          ulx=100.0,
          uly=200.0,
          lrx=300.0,
          lry=100.0,
          debug=True
      )
      """

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
    """
    Replaces specific pixel values in a GeoTIFF file using GDAL's gdal_calc.py utility.

    Parameters:
    -----------
    input_tif : str
        Path to the input GeoTIFF file.
    output_tif : str
        Path to the output GeoTIFF file where the modified data will be saved.
    src_value : int or float
        The pixel value in the input GeoTIFF that needs to be replaced.
    dst_value : int or float
        The new value to replace the src_value in the output GeoTIFF.
    debug : bool, optional (default=False)
        If True, prints debugging information, including the constructed GDAL command.

    Returns:
    --------
    None
        This function does not return anything. The modified raster is saved as the output GeoTIFF file.

    Raises:
    -------
    subprocess.CalledProcessError
        If the gdal_calc.py command fails, an error message is printed and this exception is raised.
    Exception
        If any other error occurs, it will print an error message describing the issue.

    Example:
    --------
    gdal_replace_value(
        input_tif='input_file.tif',
        output_tif='output_file.tif',
        src_value=0,
        dst_value=255,
        debug=True
    )
    """
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
    """
    Unsets the NoData value in a GeoTIFF file using GDAL's gdalwarp command.

    Parameters:
    -----------
    input_tif : str
        Path to the input GeoTIFF file.
    output_tif : str
        Path to the output GeoTIFF file where the updated data will be saved.
    debug : bool, optional (default=False)
        If True, prints debugging information, including the constructed GDAL command.

    Returns:
    --------
    None
        This function does not return anything. The modified raster is saved as the output GeoTIFF file.

    Raises:
    -------
    subprocess.CalledProcessError
        If the gdalwarp command fails, an error message is printed and this exception is raised.
    Exception
        If any other error occurs, it will print an error message describing the issue.

    Example:
    --------
    gdal_unset_nodata(
        input_tif='input_file.tif',
        output_tif='output_file.tif',
        debug=True
    )
    """
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
    """
    Resamples a GeoTIFF file to a specified resolution using GDAL's gdalwarp command.

    Parameters:
    -----------
    input_tif : str
        Path to the input GeoTIFF file.
    output_tif : str
        Path to the output GeoTIFF file where the resampled data will be saved.
    resample_alg : str
        Resampling algorithm to use (e.g., 'near', 'bilinear', 'cubic', etc.).
    x_res : float
        Target resolution in the X direction, in units of the coordinate reference system.
    y_res : float
        Target resolution in the Y direction, in units of the coordinate reference system.
    debug : bool, optional (default=False)
        If True, prints debugging information, including the input and output paths,
        as well as the constructed GDAL command.

    Returns:
    --------
    None
        This function does not return anything. The result is saved as the output GeoTIFF file.

    Raises:
    -------
    subprocess.CalledProcessError
        If the gdalwarp command fails, an error message is printed and this exception is raised.
    Exception
        If any other error occurs, it will print an error message describing the issue.

    Example:
    --------
    gdal_resample(
        input_tif='input_file.tif',
        output_tif='output_file.tif',
        resample_alg='bilinear',
        x_res=100.0,
        y_res=100.0,
        debug=True
    )
    """
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
        "-co", "BIGTIFF = YES",
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
    Sum multiple raster files and save the result to an output file using gdal_calc.py.
    NoData values are explicitly excluded to prevent them from influencing the result.

    Parameters:
    -----------
    input_files : list of str
        List of paths to the input raster files.
    output_file : str
        The path to the output file where the result will be saved.
    nodata_value : float or int, optional (default=-9999)
        The NoData value to be excluded from calculations.
    debug : bool, optional (default=False)
        If True, prints debugging information, including the constructed GDAL command.

    Returns:
    --------
    None
        This function does not return anything. The summed raster is saved to the specified output file.

    Raises:
    -------
    subprocess.CalledProcessError
        If the gdal_calc.py command fails, an error message is printed and this exception is raised.
    Exception
        If any other unexpected error occurs, it prints an error message describing the issue.

    Notes:
    ------
    - The function only supports up to 26 input raster files due to the use of alphabetic flags (-A, -B, etc.).
      If more than 26 files are provided, only the first 26 are processed.
    - Ensure that gdal_calc.py is available in your system's environment for the function to work.

    Example:
    --------
    sum_rasters(
        input_files=['raster1.tif', 'raster2.tif', 'raster3.tif'],
        output_file='sum_output.tif',
        nodata_value=-9999,
        debug=True
    )
    """
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
    Average a raster file by dividing each pixel value by a specified divisor using gdal_calc.py.
    The NoData values are excluded from the calculation to avoid skewing the result.

    Parameters:
    -----------
    input_file : str
        Path to the input raster file.
    output_file : str
        Path to the output file where the result will be saved.
    divisor : int or float, optional (default=365)
        The value to divide each pixel by to compute the average.
    nodata_value : int or float, optional (default=-9999)
        The NoData value to be excluded from calculations.
    debug : bool, optional (default=False)
        If True, prints debugging information, including the constructed GDAL command and file overwrite messages.

    Returns:
    --------
    None
        This function does not return anything. The averaged raster is saved to the specified output file.

    Raises:
    -------
    subprocess.CalledProcessError
        If the gdal_calc.py command fails, an error message is printed and this exception is raised.
    Exception
        If any other unexpected error occurs, it prints an error message describing the issue.

    Notes:
    ------
    - The function will overwrite the output file if it already exists.
    - Ensure that gdal_calc.py is available in your system's environment for the function to work.

    Example:
    --------
    average_raster(
        input_file='input_raster.tif',
        output_file='average_output.tif',
        divisor=365,
        nodata_value=-9999,
        debug=True
    )
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

    try:
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
                        resampling=Resampling.bilinear,
                        src_nodata=src.nodata,
                        dst_nodata=src.nodata
                    )

        if debug:
            print("Transformation complete.")

    except Exception as e:
        print(f"Error during transformation of {input_tif}: {e}")


def transform_folder_to_CRS(input_folder, output_folder, proj_string, debug=False):
    """
    Transforms the coordinate reference system (CRS) of a GeoTIFF file to a specified projection using rasterio.

    Parameters:
    -----------
    input_tif : str
        Path to the input GeoTIFF file that needs to be reprojected.
    output_tif : str
        Path to the output GeoTIFF file where the reprojected data will be saved.
    proj_string : str
        The projection string (e.g., EPSG code or PROJ string) for the desired output CRS.
    debug : bool, optional (default=False)
        If True, prints debugging information, including the input and output file paths and confirmation of file overwrites.

    Returns:
    --------
    None
        This function does not return anything. The transformed raster is saved to the specified output file.

    Raises:
    -------
    Exception
        If any error occurs during reading, writing, or reprojecting, an error message is printed describing the issue.

    Notes:
    ------
    - The function will overwrite the output file if it already exists.
    - Ensure that rasterio is installed and available in your environment for the function to work properly.
    - The resampling method used for the transformation is `nearest`, which may be appropriate for categorical data.

    Example:
    --------
    transform_to_CRS(
        input_tif='input_file.tif',
        output_tif='reprojected_output.tif',
        proj_string='EPSG:4326',
        debug=True
    )
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
    Extracts cluster IDs and their corresponding latitude and longitude coordinates from a shapefile.
    Also identifies and removes clusters with erroneous coordinates near (0, 0) based on a given tolerance.

    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile containing the cluster data.
    cluster_field : str
        The field name in the shapefile that represents the cluster IDs.
    lat_field : str
        The field name in the shapefile that contains the latitude values.
    lon_field : str
        The field name in the shapefile that contains the longitude values.
    tolerance : float, optional (default=1e-1)
        The tolerance value for identifying erroneous coordinates near (0, 0). Clusters within this
        tolerance range will be flagged as erroneous and excluded from the final output.

    Returns:
    --------
    tuple
        A tuple containing:
        - cluster_data : pd.DataFrame
            A DataFrame containing the valid cluster IDs along with their latitude and longitude coordinates.
        - erroneous_cluster_ids : list
            A list of cluster IDs that have been identified as having erroneous coordinates near (0, 0).

    Raises:
    -------
    ValueError
        If one or more of the specified fields do not exist in the shapefile.

    Notes:
    ------
    - The function prints the IDs and coordinates of clusters identified as erroneous.
    - The cluster IDs are converted to integers, so any non-integer values will be cast accordingly.

    Example:
    --------
    extract_cluster_data(
        shapefile_path='clusters.shp',
        cluster_field='cluster_id',
        lat_field='latitude',
        lon_field='longitude',
        tolerance=0.1
    )
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
    Converts a list of cluster coordinates from a source CRS to a destination CRS using pyproj.

    Parameters:
    -----------
    cluster_data : list of tuples
        A list of tuples, where each tuple contains (cluster_id, latitude, longitude).
    src_crs : str
        The EPSG code or PROJ string of the source coordinate reference system.
    dst_crs : str
        The EPSG code or PROJ string of the destination coordinate reference system.

    Returns:
    --------
    list of tuples
        A list of tuples, where each tuple contains (cluster_id, x, y), with 'x' and 'y' representing the coordinates in the destination CRS.

    Notes:
    ------
    - The function uses pyproj to handle coordinate transformations.
    - The input cluster IDs are cast to integers in the output, so non-integer IDs will be converted.
    - Ensure that pyproj is installed and available in your environment for the function to work properly.

    Example:
    --------
    cluster_data = [(1, 34.05, -118.25), (2, 40.7128, -74.0060)]
    converted_data = convert_cluster_coordinates(cluster_data, 'EPSG:4326', 'EPSG:3857')
    print(converted_data)
    """
    # Initialize projection objects
    src_proj = Proj(src_crs)
    dst_proj = Proj(dst_crs)

    # Create a transformer
    transformer = Transformer.from_proj(src_proj, dst_proj)

    transformed_data = []
    for cluster_id, lat, lon in cluster_data:
        # Transforming the point to projected coordinates using the transformer
        x, y = transformer.transform(lat, lon)
        transformed_data.append((int(cluster_id), x, y))
    return transformed_data


def load_raster(file_path, expected_crs, expected_pixel_size=None):
    """
    Loads a raster file and checks if its CRS and pixel size match the expected values.

    Parameters:
    -----------
    file_path : str
        Path to the raster file to be loaded.
    expected_crs : str or dict
        The expected coordinate reference system (CRS) of the raster, specified as an EPSG code or PROJ string.
    expected_pixel_size : tuple of (float, float), optional
        The expected pixel size in the format (width, height). If provided, the function checks if the raster's pixel size matches this value.

    Returns:
    --------
    tuple
        A tuple containing:
        - src : rasterio.io.DatasetReader
            The opened raster file as a rasterio dataset object.
        - crs_match : bool
            A boolean indicating whether the raster's CRS matches the expected CRS.
        - pixel_size_match : bool
            A boolean indicating whether the raster's pixel size matches the expected pixel size.
            If `expected_pixel_size` is not provided, this defaults to True.

    Notes:
    ------
    - Ensure that rasterio is installed and available in your environment for the function to work properly.
    - The CRS comparison is done using rasterio's `CRS` objects for equivalence checking.

    Example:
    --------
    src, crs_match, pixel_size_match = load_raster(
        file_path='raster.tif',
        expected_crs='EPSG:4326',
        expected_pixel_size=(30.0, 30.0)
    )
    if crs_match and pixel_size_match:
        print("The raster matches the expected CRS and pixel size.")
    """
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
    Filters a list of points to return those that are within the raster's boundaries, after transforming to a specified CRS.
    Each point must be in the format (cluster_id, x, y).

    Parameters:
    -----------
    raster : rasterio.io.DatasetReader
        The raster object which provides the bounds and CRS information.
    points : list of tuples
        A list of points, each in the format (cluster_id, x, y), where x and y are coordinates in the source CRS.
    target_crs : str
        The target coordinate reference system (CRS) to which the points and raster bounds will be transformed.
    debug : bool, optional (default=False)
        If True, prints debugging information about the transformation and filtering process.

    Returns:
    --------
    list of dict
        A list of dictionaries, each containing:
        - 'point' : shapely.geometry.Point
            A Shapely Point object representing the point's transformed coordinates.
        - 'cluster_id' : int
            The cluster ID of the point.
        Only points that fall within the raster bounds are returned.

    Notes:
    ------
    - The function transforms both raster bounds and point coordinates to the specified target CRS before performing containment checks.
    - pyproj is used for CRS transformations and Shapely is used for geometric operations.
    - Ensure that both pyproj and shapely are installed and available in your environment.

    Example:
    --------
    points = [(1, 10.0, 50.0), (2, 15.0, 55.0)]
    points_within = find_points_within_raster(raster, points, target_crs='EPSG:3035', debug=True)
    for point_data in points_within:
        print(point_data)
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
    Converts the provided x, y coordinates to the nearest vertex of the pixel in the given raster.

    Parameters:
    -----------
    raster : rasterio.io.DatasetReader
        The rasterio dataset object containing the pixel information.
    x : float
        The x coordinate in the coordinate reference system of the raster.
    y : float
        The y coordinate in the coordinate reference system of the raster.

    Returns:
    --------
    tuple of (float, float)
        The (x, y) coordinates of the nearest vertex of the pixel containing the given point.

    Notes:
    ------
    - The function first determines the pixel containing the input coordinates and then calculates the center of that pixel.
    - Based on the comparison of the input coordinates with the pixel center, the nearest vertex is computed.
    - Ensure that rasterio is installed and available in your environment for the function to work properly.

    Example:
    --------
    vertex_coords = find_nearest_vertex_rasterio(raster, x=100.5, y=200.5)
    print(f"Nearest vertex coordinates: {vertex_coords}")
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
    """
    Crops a raster into tiles centered on given points, using rasterio.

    Parameters:
    -----------
    rasterio_dataset : rasterio.io.DatasetReader
        The rasterio dataset object to be cropped.
    points : list of dict
        A list of points, where each point is represented by a dictionary containing:
        - 'point': shapely.geometry.Point object representing the coordinates.
        - 'cluster_id': Identifier for the cluster associated with the point.
    filename_prefix : str
        Prefix for the generated output filenames.
    filename_suffix : str
        Suffix for the generated output filenames.
    output_folder : str
        Path to the folder where cropped tiles will be saved.
    tile_size : int, optional (default=224)
        The size of the square tile to be cropped, in pixels.
    debug : bool, optional (default=False)
        If True, prints debugging information about the transformation and cropping process.

    Returns:
    --------
    None
        This function does not return anything. The cropped tiles are saved in the specified output folder.

    Notes:
    ------
    - The function checks if the output folder exists, and creates it if it doesn't.
    - The cropping is performed using the nearest vertex to the provided point as the center.
    - The function ensures that the crop window does not exceed the raster's boundaries.
    - If a valid window cannot be defined (e.g., the point is too close to the boundary), the crop is skipped.
    - Ensure that rasterio and shapely are installed and available in your environment.

    Example:
    --------
    points = [{'point': Point(100.5, 200.5), 'cluster_id': 1}, {'point': Point(150.0, 250.0), 'cluster_id': 2}]
    crop_raster_rasterio(
        rasterio_dataset=raster,
        points=points,
        filename_prefix='tile',
        filename_suffix='crop',
        output_folder='output_tiles',
        tile_size=224,
        debug=True
    )
    """
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
    """
    Replaces NoData values in a raster file using rasterio, and saves the updated raster to a new file.

    Parameters:
    -----------
    input_tif : str
        Path to the input GeoTIFF file.
    output_tif : str
        Path to the output GeoTIFF file where the modified data will be saved.
    dst_nodata : int or float
        The NoData value to be set in the output raster.
    src_nodata : int or float, optional
        The NoData value in the input raster to be replaced. If None, the source NoData value will not be explicitly modified.
    debug : bool, optional (default=False)
        If True, prints debugging information about the process, including the output path and NoData value settings.

    Returns:
    --------
    None
        This function does not return anything. The modified raster is saved to the specified output file.

    Notes:
    ------
    - The function reads the first band of the raster file for simplicity. If the raster has multiple bands, this will need to be adjusted.
    - The compression is set to 'deflate' with zlevel=9 to reduce output file size.
    - Ensure that rasterio is installed and available in your environment for the function to work properly.

    Example:
    --------
    rasterio_replace_nodata(
        input_tif='input_file.tif',
        output_tif='output_file.tif',
        dst_nodata=-9999,
        src_nodata=0,
        debug=True
    )
    """
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
    """
    Loads and optionally plots GeoTIFF files from a specified path, while computing and returning statistics for each file.

    Parameters:
    -----------
    file_path : str
        File path or pattern to load GeoTIFF files. Wildcards are supported for loading multiple files.
    plot_data : bool, optional (default=True)
        If True, plots the data using matplotlib.
    cmap : str, optional (default='gray')
        The colormap to use for plotting.
    precision : numpy.dtype, optional (default=np.float32)
        The data type to cast the raster data to.
    plot_geo_coords : bool, optional (default=False)
        If True, plots geographic coordinates (latitude and longitude) on the axes. Otherwise, plots pixel coordinates.
    bad_value : float, optional (default=-9999.9)
        The value considered as a 'bad' value, which will be excluded from statistics and visualization.
    clip_percentile : float, optional (default=0)
        The percentile value for clipping the data for visualization. If 0, no clipping is performed.
    figsize : tuple of (float, float), optional (default=(12, 6))
        Size of the figure for plotting.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing statistics for each loaded GeoTIFF, including:
        - File: The file name (first 10 characters).
        - Min: Minimum value of valid data.
        - Max: Maximum value of valid data.
        - Mean: Mean value of valid data.
        - Median: Median value of valid data.
        - NoData Count: Number of NoData pixels.
        - Bad Value Count: Number of pixels with bad values.
        - NoData Percent: Percentage of NoData pixels.
        - Bad Value Percent: Percentage of pixels with bad values.

    Notes:
    ------
    - NoData and bad values are masked during both statistics calculation and plotting.
    - GeoTIFF files are expected to contain a single band; adjustments may be needed for multi-band files.
    - The function modifies the colormap to visualize NoData and bad values as magenta for easier identification.
    - Ensure that rasterio, numpy, pandas, and matplotlib are installed and available in your environment.

    Example:
    --------
    stats = load_and_plot_geotiffs(
        file_path='data/*.tif',
        plot_data=True,
        cmap='viridis',
        precision=np.float32,
        plot_geo_coords=True,
        bad_value=-9999.9,
        clip_percentile=2,
        figsize=(10, 8)
    )
    print(stats)
    """
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












