import os
import subprocess

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

    # Add the output file and calculation expression to the command
    gdal_calc_command.extend(['--outfile', output_file, '--calc', '+'.join(calc_expr)])

    if debug:
        print(f"gdal_calc.py command: {' '.join(gdal_calc_command)}\n")

    try:
        # Run the command, capture output and errors, ensure it handles errors via check=True
        result = subprocess.run(gdal_calc_command, check=True, capture_output=True, text=True)

        # Success message with optional use of output
        print(f"Sum of rasters successfully saved to {output_file}")
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
        print(f"Average operation successfully saved to {output_file}")
        print("Output from command:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:

        print(f"An error occurred while summing the rasters: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def utm_zone_longitude_bounds(epsg_code):

    # Extract the UTM zone number and hemisphere from the input code
    # Extract the UTM zone number and hemisphere from the EPSG code

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
