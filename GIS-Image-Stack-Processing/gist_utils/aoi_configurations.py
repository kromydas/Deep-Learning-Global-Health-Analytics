# The latitude and longitude coordinates that define the AOI (Area of Interest) for each country
# are based on the bounding box that encompasses the entire country.
# The crs_lat and crs_lon represent the approximate center of this bounding box, rounded to the
# nearest whole degree. This simplification is intentional.
aoi_configurations = {

    'AM': {
        'countrty': 'Armenia',
        #'valid_zones': ['38N'],
        'lat_north': 41.30,
        'lat_south': 38.84,
        'lon_west': 43.45,
        'lon_east': 46.63,
        'crs_lat': 40.0,
        'crs_lon': 45.0,
        'shapefile': 'DHS/AM_2015-16_DHS_05072024_1510_211908/AMGE71FL/AMGE71FL.shp'
    },

    'TD': {
        'countrty': 'Chad',
        #'valid_zones': ['33N', '34N'],
        'lat_north': 23.41,
        'lat_south': 7.44,
        'lon_west': 13.47,
        'lon_east': 24.00,
        'crs_lat': 14.0,
        'crs_lon': 18.0,
        'shapefile': 'DHS/TD_2014-15_DHS_05072024_1511_211908/TDGE71FL/TDGE71FL.shp'
    },

    'IN': {
        'countrty': 'India',
        #'valid_zones': ['42N', '43N', '44N', '45N', '46N', '47N'],
        'lat_north': 35.50,
        'lat_south': 6.75,
        'lon_west': 68.16,
        'lon_east': 97.40,
        'crs_lat': 21.0,
        'crs_lon': 83.0,
        'shapefile': 'DHS/IN_2019-21/IAGE7AFL/IAGE7AFL.shp'
    },

    'JO': {
        'countrty': 'Jordan',
        #'valid_zones': ['36N', '37N'],
        'lat_north': 33.37,
        'lat_south': 29.19,
        'lon_west': 34.95,
        'lon_east': 39.30,
        'crs_lat': 31.0,
        'crs_lon': 37.0,
        'shapefile': 'DHS/JO_2017-18_DHS_05072024_1514_211908/JOGE71FL/JOGE71FL.shp'
    },

    'ML': {
        'countrty': 'Mali',
        #'valid_zones': ['29N', '30N', '31N'],
        'lat_north': 25.00,
        'lat_south': 10.16,
        'lon_west': -12.24,
        'lon_east': 4.24,
        'crs_lat': 18.0,
        'crs_lon': -4.0,
        'shapefile': 'DHS/ML_2018_DHS_05072024_1516_211908/MLGE7AFL/MLGE7AFL.shp'
    },

    'MR': {
        'countrty': 'Mauritania',
        #'valid_zones': ['28N', '29N', '30N'],
        'lat_north': 27.29,
        'lat_south': 14.72,
        'lon_west': -17.07,
        'lon_east': -4.83,
        'crs_lat': 21.0,
        'crs_lon': -11.0,
        'shapefile': 'DHS/MA_2003-04_DHS_05072024_1622_211908/MAGE43FL/MAGE43FL.shp'
    },

    'MB': {
        'countrty': 'Moldova',
        #'valid_zones': ['35N'],
        'lat_north': 48.49,
        'lat_south': 45.47,
        'lon_west': 26.61,
        'lon_east': 30.13,
        'crs_lat': 47.0,
        'crs_lon': 28.0,
        'shapefile': 'DHS/MB_2005_DHS_05072024_1518_211908/MBGE52FL/MBGE52FL.shp'
    },

    'MA': {
        'countrty': 'Morocco',
        #'valid_zones': ['28N', '29N', '30N'],
        'lat_north': 35.93,
        'lat_south': 21.33,
        'lon_west': -17.02,
        'lon_east': -1.02,
        'crs_lat': 29.0,
        'crs_lon': -9.0,
        'shapefile': 'DHS/MR_2019-21_DHS_05072024_1516_211908/MRGE71FL/MRGE71FL.shp'
    },

    'NI': {
        'countrty': 'Niger',
        #'valid_zones': ['31N', '32N', '33N'],
        'lat_north': 23.52,
        'lat_south': 11.69,
        'lon_west': 0.169,
        'lon_east': 16.00,
        'crs_lat': 18.0,
        'crs_lon': 8.0,
        'shapefile': 'DHS/NI_2012_DHS_05072024_1623_211908/NIGE61FL/NIGE61FL.shp'
    },

    'PK': {
        'countrty': 'Pakistan',
        #'valid_zones': ['41N', '42N', '43N'],
        'lat_north': 37.08,
        'lat_south': 23.63,
        'lon_west': 60.87,
        'lon_east': 77.84,
        'crs_lat': 30.0,
        'crs_lon': 69.0,
        'shapefile': 'DHS/PK_2017-18_DHS_05072024_2158_211908/PKGE71FL/PKGE71FL.shp'
    },

    'SN': {
        'countrty': 'Senegal',
        #'valid_zones': ['28N', '29N'],
        'lat_north': 16.69,
        'lat_south': 12.31,
        'lon_west': -17.54,
        'lon_east': -11.36,
        'crs_lat': 14.0,
        'crs_lon': 14.0,
        'shapefile': 'DHS/SN_2023_CONTINUOUSDHS_07122024_423_211908/SNGE8RFL/SNGE8RFL.shp'
    }
}