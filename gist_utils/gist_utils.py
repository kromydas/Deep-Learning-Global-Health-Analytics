
import subprocess

def run_gdalinfo(tif_path):
    # Construct the command string
    command = f"gdalinfo -mm {tif_path}"

    # Execute the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


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
    subprocess.run(gdal_translate_command)


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

    if src_nodata is not None:
        gdal_warp_command.extend(["-srcnodata", str(src_nodata)])

    gdal_warp_command.extend(["-dstnodata", str(dst_nodata)])

    if debug:
        print(f"gdalwarp command: {' '.join(gdal_warp_command)}\n")

    # Run the gdalwarp command
    subprocess.run(gdal_warp_command, check=True)

    if debug:
        print(f"Output TIF has been written to: {output_tif}")


def gdal_resample(input_tif, output_tif, x_res, y_res, debug=False):
    if debug:
        # Debugging prints to check the file paths
        print(f"Input  TIF: {input_tif}")
        print(f"Output TIF: {output_tif}\n")

    gdalwarp_command = [
        "gdalwarp",
        "-overwrite",
        "-tr", str(x_res), str(y_res),
        "-r", "bilinear",
        "-co", "COMPRESS=DEFLATE",
        "-co", "ZLEVEL=9",
        input_tif,
        output_tif
    ]
    subprocess.run(gdalwarp_command)


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

    # Construct the gdalwarp command
    gdalwarp_command = [
        "gdalwarp",
        "-t_srs", utm_zone,  # Target SRS
        "-co", "COMPRESS=DEFLATE",
        "-co", "ZLEVEL=9",
        input_tif,
        output_tif
    ]

    # Execute the gdalwarp command
    subprocess.run(gdalwarp_command)


