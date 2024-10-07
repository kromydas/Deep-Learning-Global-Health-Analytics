# The latitude and longitude coordinates that define the AOI (Area of Interest) for each country
# are based on the bounding box that encompasses the entire country.
# The crs_lat and crs_lon represent the approximate center of this bounding box, rounded to the
# nearest whole degree. This simplification is intentional.
aoi_configurations = {

    'AM': {
        'countrty': 'Armenia',
        'lat_north': 41.30,
        'lat_south': 38.84,
        'lon_west': 43.45,
        'lon_east': 46.63,
        'crs_lat': 40.0,
        'crs_lon': 45.0,
        'shapefile': 'DHS/AM_2015-16_DHS_05072024_1510_211908/AMGE71FL/AMGE71FL.shp',
        'recode_hr': 'DHS/AM_2015-16_DHS_05072024_1510_211908/AMHR72SV/AMHR72FL.SAV',
        'recode_kr': 'DHS/AM_2015-16_DHS_05072024_1510_211908/AMKR72SV/AMKR72FL.SAV',
        'recode_br': 'DHS/AM_2015-16_DHS_05072024_1510_211908/AMBR72SV/AMBR72FL.SAV'
    },

    'TD': {
        'countrty': 'Chad',
        'lat_north': 23.41,
        'lat_south': 7.44,
        'lon_west': 13.47,
        'lon_east': 24.00,
        'crs_lat': 14.0,
        'crs_lon': 18.0,
        'shapefile': 'DHS/TD_2014-15_DHS_05072024_1511_211908/TDGE71FL/TDGE71FL.shp',
        'recode_hr': 'DHS/TD_2014-15_DHS_05072024_1511_211908/TDHR71SV/TDHR71FL.SAV',
        'recode_kr': 'DHS/TD_2014-15_DHS_05072024_1511_211908/TDKR71SV/TDKR71FL.SAV',
        'recode_br': 'DHS/TD_2014-15_DHS_05072024_1511_211908/TDBR71SV/TDBR71FL.SAV'
    },

    'IN': {
        'countrty': 'India',
        'lat_north': 35.50,
        'lat_south': 6.75,
        'lon_west': 68.16,
        'lon_east': 97.40,
        'crs_lat': 21.0,
        'crs_lon': 83.0,
        'shapefile': 'DHS/IN_2019-21/IAGE7AFL/IAGE7AFL.shp',
        'recode_hr': 'DHS/IN_2019-21/IAHR7ESV/IAHR7EFL.SAV',
        'recode_kr': 'DHS/IN_2019-21/IAKR7ESV/IAKR7EFL.SAV',
        'recode_br': 'DHS/IN_2019-21/IABR7ESV/IABR7EFL.SAV'
    },

    'JO': {
        'countrty': 'Jordan',
        'lat_north': 33.37,
        'lat_south': 29.19,
        'lon_west': 34.95,
        'lon_east': 39.30,
        'crs_lat': 31.0,
        'crs_lon': 37.0,
        'shapefile': 'DHS/JO_2017-18_DHS_05072024_1514_211908/JOGE71FL/JOGE71FL.shp',
        'recode_hr': 'DHS/JO_2017-18_DHS_05072024_1514_211908/JOHR74SV/JOHR74FL.SAV',
        'recode_kr': 'DHS/JO_2017-18_DHS_05072024_1514_211908/JOKR74SV/JOKR74FL.SAV',
        'recode_br': 'DHS/JO_2017-18_DHS_05072024_1514_211908/JOBR74SV/JOBR74FL.SAV'
    },

    'ML': {
        'countrty': 'Mali',
        'lat_north': 25.00,
        'lat_south': 10.16,
        'lon_west': -12.24,
        'lon_east': 4.24,
        'crs_lat': 18.0,
        'crs_lon': -4.0,
        'shapefile': 'DHS/ML_2018_DHS_05072024_1516_211908/MLGE7AFL/MLGE7AFL.shp',
        'recode_hr': 'DHS/ML_2018_DHS_05072024_1516_211908/MLHR7ASV/MLHR7AFL.SAV',
        'recode_kr': 'DHS/ML_2018_DHS_05072024_1516_211908/MLKR7ASV/MLKR7AFL.SAV',
        'recode_br': 'DHS/ML_2018_DHS_05072024_1516_211908/MLBR7ASV/MLBR7AFL.SAV'
    },

    'MR': {
        'countrty': 'Mauritania',
        'lat_north': 27.29,
        'lat_south': 14.72,
        'lon_west': -17.07,
        'lon_east': -4.83,
        'crs_lat': 21.0,
        'crs_lon': -11.0,
        'shapefile': 'DHS/MR_2019-21_DHS_05072024_1516_211908/MRGE71FL/MRGE71FL.shp',
        'recode_hr': 'DHS/MR_2019-21_DHS_05072024_1516_211908/MRHR71SV/MRHR71FL.SAV',
        'recode_kr': 'DHS/MR_2019-21_DHS_05072024_1516_211908/MRKR71SV/MRKR71FL.SAV',
        'recode_br': 'DHS/MR_2019-21_DHS_05072024_1516_211908/MRBR71SV/MRBR71FL.SAV'
    },

    'MB': {
        'countrty': 'Moldova',
        'lat_north': 48.49,
        'lat_south': 45.47,
        'lon_west': 26.61,
        'lon_east': 30.13,
        'crs_lat': 47.0,
        'crs_lon': 28.0,
        'shapefile': 'DHS/MB_2005_DHS_05072024_1518_211908/MBGE52FL/MBGE52FL.shp',
        'recode_hr': 'DHS/MB_2005_DHS_05072024_1518_211908/MBHR53SV/MBHR53FL.SAV',
        'recode_kr': 'DHS/MB_2005_DHS_05072024_1518_211908/MBKR53SV/MBKR53FL.SAV',
        'recode_br': 'DHS/MB_2005_DHS_05072024_1518_211908/MBBR53SV/MBBR53FL.SAV'
    },

    'MA': {
        'countrty': 'Morocco',
        'lat_north': 35.93,
        'lat_south': 21.33,
        'lon_west': -17.02,
        'lon_east': -1.02,
        'crs_lat': 29.0,
        'crs_lon': -9.0,
        'shapefile': 'DHS/MA_2003-04_DHS_05072024_1622_211908/MAGE43FL/MAGE43FL.shp',
        'recode_hr': 'DHS/MA_2003-04_DHS_05072024_1622_211908/MAHR43SV/MAHR43FL.SAV',
        'recode_kr': 'DHS/MA_2003-04_DHS_05072024_1622_211908/MAKR43SV/MAKR43FL.SAV',
        'recode_br': 'DHS/MA_2003-04_DHS_05072024_1622_211908/MABR43SV/MABR43FL.SAV'
    },

    'NI': {
        'countrty': 'Niger',
        'lat_north': 23.52,
        'lat_south': 11.69,
        'lon_west': 0.169,
        'lon_east': 16.00,
        'crs_lat': 18.0,
        'crs_lon': 8.0,
        'shapefile': 'DHS/NI_2012_DHS_05072024_1623_211908/NIGE61FL/NIGE61FL.shp',
        'recode_hr': 'DHS/NI_2012_DHS_05072024_1623_211908/NIHR61SV/NIHR61FL.SAV',
        'recode_kr': 'DHS/NI_2012_DHS_05072024_1623_211908/NIKR61SV/NIKR61FL.SAV',
        'recode_br': 'DHS/NI_2012_DHS_05072024_1623_211908/NIBR61SV/NIBR61FL.SAV'
    },

    'PK': {
        'countrty': 'Pakistan',
        'lat_north': 37.08,
        'lat_south': 23.63,
        'lon_west': 60.87,
        'lon_east': 77.84,
        'crs_lat': 30.0,
        'crs_lon': 69.0,
        'shapefile': 'DHS/PK_2017-18_DHS_05072024_2158_211908/PKGE71FL/PKGE71FL.shp',
        'recode_hr': 'DHS/PK_2017-18_DHS_05072024_2158_211908/PKHR71SV/PKHR71FL.SAV',
        'recode_kr': 'DHS/PK_2017-18_DHS_05072024_2158_211908/PKKR71SV/PKKR71FL.SAV',
        'recode_br': 'DHS/PK_2017-18_DHS_05072024_2158_211908/PKBR71SV/PKBR71FL.SAV'
    },

    'SN': {
        'countrty': 'Senegal',
        'lat_north': 16.69,
        'lat_south': 12.31,
        'lon_west': -17.54,
        'lon_east': -11.36,
        'crs_lat': 14.0,
        'crs_lon': -14.0,
        'shapefile': 'DHS/SN_2019_CONTINUOUSDHS_05072024_1625_211908/SNGE8BFL/SNGE8BFL.shp',
        'recode_hr': 'DHS/SN_2019_CONTINUOUSDHS_05072024_1625_211908/SNHR8BSV/SNHR8BFL.SAV',
        'recode_kr': 'DHS/SN_2019_CONTINUOUSDHS_05072024_1625_211908/SNKR8BSV/SNKR8BFL.SAV',
        'recode_br': 'DHS/SN_2019_CONTINUOUSDHS_05072024_1625_211908/SNBR8BSV/SNBR8BFL.SAV'
    }
}