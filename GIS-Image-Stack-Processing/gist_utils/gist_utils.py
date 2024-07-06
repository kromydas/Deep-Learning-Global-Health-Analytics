import os
import shutil
import subprocess
import geopandas as gpd
from osgeo import gdal, ogr, osr

from pyproj import Transformer
import rasterio
from shapely.geometry import Point, Polygon
from rasterio.windows import Window

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

def gdal_replace_nodata(input_tif, output_tif, dst_nodata, src_nodata=None, debug=False):

    if debug:
        print(f"Input TIF: {input_tif}")
        print(f"Final Output TIF: {output_tif}\n")

    # Construct the gdalwarp command
    gdal_warp_command = [
        "gdalwarp",
        "-overwrite",
        "-co", "COMPRESS=DEFLATE",
        "-co", "ZLEVEL=9",
        input_tif,
        output_tif
    ]

    if debug:
        print(f"gdalwarp command: {' '.join(gdal_warp_command)}\n")

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

def sum_rasters(input_files, output_file, debug=False):
    """
    Sum a list of raster files using gdal_calc.py and save the result to a specified output file.

    Args:
        input_files (list of str): List of paths to the input raster files.
        output_file (str): The path to the output file where the result will be saved.
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
            calc_expr.append(letter)
        else:
            print("Warning: More than 26 files detected; only the first 26 will be processed.")
            break

    # Adding the overwrite option
    gdal_calc_command.extend(['--outfile', output_file, '--calc', '+'.join(calc_expr), '--overwrite'])

    if debug:
        print(f"gdal_calc.py command: {' '.join(gdal_calc_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdal_calc_command, check=True, capture_output=True, text=True)

        # Success message with optional use of output
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

def average_raster(input_file, output_file, divisor=365, debug=False):
    """
    Average a raster file by dividing each pixel by a specified divisor using gdal_calc.py.

    Args:
    input_file (str): Path to the input raster file.
    output_file (str): Path to the output file where the result will be saved.
    divisor (int or float): The value to divide each pixel by.
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
        '--calc', f"A/{divisor}"
    ]

    if debug:
        print(f"gdal_calc.py command: {' '.join(gdal_calc_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdal_calc_command, check=True, capture_output=True, text=True)

        # Success message with optional use of output
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

def convert_cluster_coordinates(cluster_data, src_crs, dst_crs):
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

def load_raster(file_path, expected_crs, expected_pixel_size=None):
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


def extract_latitudes_from_raster_utm(raster):

    # Define the transformer to convert UTM to WGS 84 (lat, lon)
    transformer = Transformer.from_crs(raster.crs, 'EPSG:4326', always_xy=True)

    # Get the bounds of the raster
    bounds = raster.bounds

    # Transform the coordinates
    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)

    return max_lat, min_lat

def find_points_within_raster_zone(raster, points, debug=False):
    """
    Filters a list of points, returning those within the raster's UTM zone boundaries.
    Since the input raster has been extended beyond the UTM zone, we won't use the actual raster
    bounds to determine inclusion, but rather the UTM for the specified zone.
    All points must be tuples in the format (cluster_id, x, y).

    Parameters:
        raster (rasterio.io.DatasetReader): The raster object in UTM which provides the bounds.
        points (list): A list of points in the format (cluster_id, x, y), where x and y are coordinates.

    Returns:
        list: A list of dictionaries, each containing a 'point' (Shapely Point object) and 'cluster_id',
              for points that are within the contracted bounds of the raster.
    """

    # Retrieve raster CRS to get UTM Zone longitude bands.
    epsg_code = f"EPSG:{raster.crs.to_epsg()}"
    lon_west, lon_east = utm_zone_longitude_bounds(epsg_code)

    lat_north, lat_south = extract_latitudes_from_raster_utm(raster)

    ul_x, ul_y = latlon_to_utm(lat_north, lon_west, dst_crs=epsg_code)
    ur_x, ur_y = latlon_to_utm(lat_north, lon_east, dst_crs=epsg_code)
    ll_x, ll_y = latlon_to_utm(lat_south, lon_west, dst_crs=epsg_code)
    lr_x, lr_y = latlon_to_utm(lat_south, lon_east, dst_crs=epsg_code)

    ul = (ul_x, ul_y)
    ur = (ur_x, ur_y)
    lr = (lr_x, lr_y)
    ll = (ll_x, ll_y)

    aoi_bounds = Polygon([ul, ur, lr, ll])

    coordinates = list(aoi_bounds.exterior.coords)
    if debug:
        for x, y in coordinates:
            lat, lon = utm_to_latlon(x, y)
            print(f"bbox: ({lat:.3f} {lon:.3f})")

    # Filter points based on whether they lie within the contracted bounds
    points_within_raster = []
    for cluster_id, x, y in points:
        point = Point(x, y)
        if debug:
            print(f"Checking point {point} with cluster_id {cluster_id}")  # Debug print each point and its check
        if aoi_bounds.contains(point):
            points_within_raster.append({'point': point, 'cluster_id': cluster_id})
            if debug:
                print(f"Point {point} is within the contracted bounds.")  # Debug success inclusion
        else:
            if debug:
                print(f"Point {point} is NOT within the contracted bounds.\n")  # Debug failure exclusion

    return points_within_raster

def find_nearest_vertex_rasterio(raster, x, y):
    """
    Convert UTM coordinates to the nearest vertex of the pixel using.

    Parameters:
        raster (rasterio.io.DatasetReader): The rasterio dataset object.
        x (float): X coordinate in UTM.
        y (float): Y coordinate in UTM.

    Returns:
        tuple: The UTM coordinates of the nearest vertex of a pixel.
    """
    # Get the pixel coordinates of the center of the pixel that contains the point
    px, py = raster.index(x, y)

    # Get the UTM coordinates for the center of this pixel
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

def find_nearest_vertex_rasterio(rasterio_dataset, x, y):
    """
    Convert UTM coordinates to the nearest vertex of the pixel using rasterio.

    Parameters:
        rasterio_dataset (rasterio.io.DatasetReader): The rasterio dataset.
        x (float): X coordinate in UTM.
        y (float): Y coordinate in UTM.

    Returns:
        tuple: The UTM coordinates of the nearest vertex of a pixel.
    """
    # Convert from UTM coordinates to pixel coordinates
    px, py = rasterio_dataset.index(x, y)

    # Calculate the UTM coordinates for the nearest vertex (pixel corner)
    vertex_x, vertex_y = rasterio_dataset.xy(px, py, offset='ul')

    return (vertex_x, vertex_y)

def crop_raster_rasterio(rasterio_dataset, points, filename_prefix, filename_suffix, output_folder, tile_size=224, debug=False):
    """
    Crop raster based on provided UTM coordinates and saves the resulting tiles into the specified folder. The function
    uses the rasterio library to handle the cropping process.

    Parameters:
        rasterio_dataset (rasterio.DatasetReader): The raster dataset from which to crop tiles.
        points (list of dicts): A list of dictionaries, each containing a 'point' (Shapely Point object with UTM coordinates)
                                and 'cluster_id' which identifies the group or cluster the point belongs to.
        filename_prefix (str): Prefix for the output filenames.
        filename_suffix (str): Suffix for the output filenames.
        output_folder (str): The directory path where the cropped tiles will be saved.
        tile_size (int, optional): The size (width and height) of each square tile in pixels. Default is 224 pixels.
        debug (bool, optional): If set to True, prints debug statements including transformation details and pixel calculations.

    Returns:
        None: This function does not return a value but writes cropped raster tiles to disk.

    Example Usage:
        >>> crop_raster_rasterio(raster_dataset, points, 'tile', '2024', './output_tiles')

    Notes:
        Each point in 'points' should be a dictionary with a 'point' key containing a Shapely Point object and
        a 'cluster_id' key. The 'point' should have x (longitude) and y (latitude) coordinates in UTM.
        The function skips any item in 'points' that does not match the expected format.
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

        center_utm_x, center_utm_y = find_nearest_vertex_rasterio(rasterio_dataset, point.x, point.y)

        if debug:
           print(f"Converted UTM ({point.x}, {point.y}) to Vertex ({center_utm_x}, {center_utm_y})\n")

        # Convert UTM coordinates back to pixel coordinates for cropping
        py, px = rasterio_dataset.index(center_utm_x, center_utm_y)
        px_ul_x = px - tile_size // 2
        px_ul_y = py - tile_size // 2

        if debug:
           print(f"Converted UTM ({point.x}, {point.y}) to Pixel ({px}, {py})\n")

        if debug:
            print(f"Calculated upper-left corner of the crop: px_ul_x={px_ul_x}, px_ul_y={px_ul_y}\n")

        window = rasterio.windows.Window(px_ul_x, px_ul_y, tile_size, tile_size)
        transform = rasterio_dataset.window_transform(window)

        output_filename = f"{filename_prefix}_{idx:03d}_C-{cluster_id:03d}_{filename_suffix}.tif"
        output_raster_path = os.path.join(output_folder, output_filename)

        # Read the window and write it to a new file
        with rasterio.open(
            output_raster_path,
            'w',
            driver='GTiff',
            height=tile_size,
            width=tile_size,
            count=rasterio_dataset.count,
            dtype=rasterio_dataset.dtypes[0],
            crs=rasterio_dataset.crs,
            transform=transform
        ) as dst:
            dst.write(rasterio_dataset.read(window=window, boundless=True, fill_value=0))

    print(f"Crops are saved in {output_folder}")
