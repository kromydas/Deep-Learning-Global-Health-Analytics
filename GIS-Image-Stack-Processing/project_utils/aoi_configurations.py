import json
import pandas as pd
#--------------------------------------------------------------------------------------------------
# The aoi_configurations dictionary stores AOI-level (Area of Interest) data for each country,
# using the DHS two-letter country code as the dictionary key. Each entry contains the following
# information:
#
#     Country name
#     Bounding box coordinates for the AOI (latitude and longitude values)
#     Coordinates for the center of the AOI's CRS (Coordinate Reference System)
#     Path names for the DHS shapefile that contains the coordinates for each DHS cluster
#     Path names for the household and child recode files (the birth recode file is included
#                for future reference but is not currently used)
#
# Important Notes:
# The latitude and longitude coordinates that define the AOI are based on a bounding box that
# encompasses the entire country.
# The crs_lat and crs_lon fields represent the approximate center of this bounding box, rounded to
# the nearest whole degree. This simplification is intentional for ease of use.
#--------------------------------------------------------------------------------------------------
aoi_configurations = {

    'AM': {
        'countrty': 'Armenia',
        'lat_north': 41.30,
        'lat_south': 38.84,
        'lon_west': 43.45,
        'lon_east': 46.63,
        'crs_lat': 40.0,
        'crs_lon': 45.0,
        'shapefile': 'DHS/AM_2015-16_DHS/AMGE71FL/AMGE71FL.shp',
        'recode_hr': 'DHS/AM_2015-16_DHS/AMHR72SV/AMHR72FL.SAV',
        'recode_kr': 'DHS/AM_2015-16_DHS/AMKR72SV/AMKR72FL.SAV',
        'recode_br': 'DHS/AM_2015-16_DHS/AMBR72SV/AMBR72FL.SAV'
    },

    'TD': {
        'countrty': 'Chad',
        'lat_north': 23.41,
        'lat_south': 7.44,
        'lon_west': 13.47,
        'lon_east': 24.00,
        'crs_lat': 14.0,
        'crs_lon': 18.0,
        'shapefile': 'DHS/TD_2014-15_DHS/TDGE71FL/TDGE71FL.shp',
        'recode_hr': 'DHS/TD_2014-15_DHS/TDHR71SV/TDHR71FL.SAV',
        'recode_kr': 'DHS/TD_2014-15_DHS/TDKR71SV/TDKR71FL.SAV',
        'recode_br': 'DHS/TD_2014-15_DHS/TDBR71SV/TDBR71FL.SAV'
    },

    'JO': {
        'countrty': 'Jordan',
        'lat_north': 33.37,
        'lat_south': 29.19,
        'lon_west': 34.95,
        'lon_east': 39.30,
        'crs_lat': 31.0,
        'crs_lon': 37.0,
        'shapefile': 'DHS/JO_2017-18_DHS/JOGE71FL/JOGE71FL.shp',
        'recode_hr': 'DHS/JO_2017-18_DHS/JOHR74SV/JOHR74FL.SAV',
        'recode_kr': 'DHS/JO_2017-18_DHS/JOKR74SV/JOKR74FL.SAV',
        'recode_br': 'DHS/JO_2017-18_DHS/JOBR74SV/JOBR74FL.SAV'
    },

    'ML': {
        'countrty': 'Mali',
        'lat_north': 25.00,
        'lat_south': 10.16,
        'lon_west': -12.24,
        'lon_east': 4.24,
        'crs_lat': 18.0,
        'crs_lon': -4.0,
        'shapefile': 'DHS/ML_2018_DHS/MLGE7AFL/MLGE7AFL.shp',
        'recode_hr': 'DHS/ML_2018_DHS/MLHR7ASV/MLHR7AFL.SAV',
        'recode_kr': 'DHS/ML_2018_DHS/MLKR7ASV/MLKR7AFL.SAV',
        'recode_br': 'DHS/ML_2018_DHS/MLBR7ASV/MLBR7AFL.SAV'
    },

    'MR': {
        'countrty': 'Mauritania',
        'lat_north': 27.29,
        'lat_south': 14.72,
        'lon_west': -17.07,
        'lon_east': -4.83,
        'crs_lat': 21.0,
        'crs_lon': -11.0,
        'shapefile': 'DHS/MR_2019-21_DHS/MRGE71FL/MRGE71FL.shp',
        'recode_hr': 'DHS/MR_2019-21_DHS/MRHR71SV/MRHR71FL.SAV',
        'recode_kr': 'DHS/MR_2019-21_DHS/MRKR71SV/MRKR71FL.SAV',
        'recode_br': 'DHS/MR_2019-21_DHS/MRBR71SV/MRBR71FL.SAV'
    },

    'MB': {
        'countrty': 'Moldova',
        'lat_north': 48.49,
        'lat_south': 45.47,
        'lon_west': 26.61,
        'lon_east': 30.13,
        'crs_lat': 47.0,
        'crs_lon': 28.0,
        'shapefile': 'DHS/MB_2005_DHS/MBGE52FL/MBGE52FL.shp',
        'recode_hr': 'DHS/MB_2005_DHS/MBHR53SV/MBHR53FL.SAV',
        'recode_kr': 'DHS/MB_2005_DHS/MBKR53SV/MBKR53FL.SAV',
        'recode_br': 'DHS/MB_2005_DHS/MBBR53SV/MBBR53FL.SAV'
    },

    'MA': {
        'countrty': 'Morocco',
        'lat_north': 35.93,
        'lat_south': 21.33,
        'lon_west': -17.02,
        'lon_east': -1.02,
        'crs_lat': 29.0,
        'crs_lon': -9.0,
        'shapefile': 'DHS/MA_2003-04_DHS/MAGE43FL/MAGE43FL.shp',
        'recode_hr': 'DHS/MA_2003-04_DHS/MAHR43SV/MAHR43FL.SAV',
        'recode_kr': 'DHS/MA_2003-04_DHS/MAKR43SV/MAKR43FL.SAV',
        'recode_br': 'DHS/MA_2003-04_DHS/MABR43SV/MABR43FL.SAV'
    },

    'NI': {
        'countrty': 'Niger',
        'lat_north': 23.52,
        'lat_south': 11.69,
        'lon_west': 0.169,
        'lon_east': 16.00,
        'crs_lat': 18.0,
        'crs_lon': 8.0,
        'shapefile': 'DHS/NI_2012_DHS/NIGE61FL/NIGE61FL.shp',
        'recode_hr': 'DHS/NI_2012_DHS/NIHR61SV/NIHR61FL.SAV',
        'recode_kr': 'DHS/NI_2012_DHS/NIKR61SV/NIKR61FL.SAV',
        'recode_br': 'DHS/NI_2012_DHS/NIBR61SV/NIBR61FL.SAV'
    },

    'PK': {
        'countrty': 'Pakistan',
        'lat_north': 37.08,
        'lat_south': 23.63,
        'lon_west': 60.87,
        'lon_east': 77.84,
        'crs_lat': 30.0,
        'crs_lon': 69.0,
        'shapefile': 'DHS/PK_2017-18_DHS/PKGE71FL/PKGE71FL.shp',
        'recode_hr': 'DHS/PK_2017-18_DHS/PKHR71SV/PKHR71FL.SAV',
        'recode_kr': 'DHS/PK_2017-18_DHS/PKKR71SV/PKKR71FL.SAV',
        'recode_br': 'DHS/PK_2017-18_DHS/PKBR71SV/PKBR71FL.SAV'
    },

    'SN': {
        'countrty': 'Senegal',
        'lat_north': 16.69,
        'lat_south': 12.31,
        'lon_west': -17.54,
        'lon_east': -11.36,
        'crs_lat': 14.0,
        'crs_lon': -14.0,
        'shapefile': 'DHS/SN_2019_CONTINUOUSDHS/SNGE8BFL/SNGE8BFL.shp',
        'recode_hr': 'DHS/SN_2019_CONTINUOUSDHS/SNHR8BSV/SNHR8BFL.SAV',
        'recode_kr': 'DHS/SN_2019_CONTINUOUSDHS/SNKR8BSV/SNKR8BFL.SAV',
        'recode_br': 'DHS/SN_2019_CONTINUOUSDHS/SNBR8BSV/SNBR8BFL.SAV'
        # ------------------------------------------------------------------------------
        # More recent data is available for Senegal (2023), but it does not contain .SAV
        # files. The code that processes the recode files would need to be generalized
        # further to handle .SAS files before the 2023 data can be utilized.
        # ------------------------------------------------------------------------------
        # 'shapefile': 'DHS/SN_2023_CONTINUOUSDHS/SNGE8RFL/SNGE8RFL.shp',
        # 'recode_hr': 'DHS/SN_2023_CONTINUOUSDHS/SNHR8RSV/SNHR8RFL.SAV',
        # 'recode_kr': 'DHS/SN_2023_CONTINUOUSDHS/SNKR8RSV/SNKR8RFL.SAV',
        # 'recode_br': 'DHS/SN_2023_CONTINUOUSDHS/SNBR8RSV/SNBR8RFL.SAV'
    }
}


def process_aoi_target_json(aoi_target_json_path, country_code):
    """
    Processes an AOI target JSON file into DataFrames for DHS and geospatial analysis.

    Args:
        aoi_target_json_path (str): Path to the JSON file containing AOI target data.
        country_code (str): Country code to be added to the DataFrame.

    Returns:
        tuple: A tuple containing:
            - dhs_df (pd.DataFrame): DataFrame with 'cluster_id' and target data.
            - geospatial_df (pd.DataFrame): DataFrame indexed by 'cluster_id' for geospatial matching.
    """
    # Load the JSON file into a Python dictionary
    with open(aoi_target_json_path, 'r') as f:
        data_dict = json.load(f)

    # Remove the 'metadata' entry if present
    if 'metadata' in data_dict:
        data_dict.pop('metadata')

    # Remove any non-cluster keys (such as 'clusters', if needed)
    if 'clusters' in data_dict:
        data_dict = data_dict['clusters']  # Access the actual cluster data under the 'clusters' key

    # Convert the remaining dictionary to a DataFrame
    dhs_df = pd.DataFrame.from_dict(data_dict, orient='index')

    # Ensure that the 'cluster_id' column is an integer
    dhs_df.index = dhs_df.index.astype(int)

    # Reset the index to make 'cluster_id' a column and rename 'index' column
    dhs_df = dhs_df.reset_index().rename(columns={'index': 'cluster_id'})

    # Sort the DataFrame by 'cluster_id' to ensure correct numerical ordering
    dhs_df = dhs_df.sort_values(by='cluster_id')

    # Add the 'country_code' column
    dhs_df['country_code'] = country_code

    # Copy dhs_df to geospatial_df
    geospatial_df = dhs_df.copy()

    # Ensure 'cluster_id' is the index for proper matching in geospatial_df
    geospatial_df.set_index('cluster_id', inplace=True)

    # Display debug information
    print(dhs_df.head())
    print('Number of records in dhs_df: ', len(dhs_df))
    print("\n")
    print(geospatial_df.head())
    print('Number of records in geospatial_df: ', len(geospatial_df))

    return dhs_df, geospatial_df