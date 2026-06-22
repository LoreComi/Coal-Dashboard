"""Configuration — Coal Desk Weather / CDD Report.

Cities, regions, populations, constants.
"""
from __future__ import annotations

# CDD parameters
BASE_TEMP = 18.0  # Celsius
SEASON_START_MONTH = 4
SEASON_START_DAY = 15
HIST_START_YEAR = 2000
HIST_END_YEAR = 2024

# Source tables — Temperature
TABLE_HIST = "dna_prod_silver.meteomatics.temperature"
TABLE_FCST = "dna_prod_silver.meteomatics.temperature_forecast"
CURVE_HIST = "t_mean_2m_24h_c_ecmwf_era5_p1d"
CURVE_FCST = "t_mean_2m_24h_c_ecmwf_ens_p1d"
MODEL_HIST = "ecmwf-era5"
MODEL_FCST = "ecmwf-ens"

# Source tables — Precipitation
TABLE_PRECIP_HIST = "dna_prod_silver.meteomatics.precipitation"
TABLE_PRECIP_FCST = "dna_prod_silver.meteomatics.precipitation_forecast"
CURVE_PRECIP_HIST = "precip_24h_mm_ecmwf_era5_p1d"
CURVE_PRECIP_FCST = "precip_24h_mm_ecmwf_ens_p1d"

# City coordinates (rounded to 0.5 deg grid)
CITY_LOCATIONS: dict[str, dict] = {
    'Augsburg': {'latitude': 48.5, 'longitude': 11.0},
    'Karlsruhe': {'latitude': 49.0, 'longitude': 8.5},
    'Mannheim': {'latitude': 49.5, 'longitude': 8.5},
    'Bergerac': {'latitude': 45.0, 'longitude': 0.5},
    'Le Mans': {'latitude': 48.0, 'longitude': 0.0},
    'Delhi': {'latitude': 29.0, 'longitude': 77.0},
    'Jamnagar': {'latitude': 22.5, 'longitude': 70.0},
    'Haldia': {'latitude': 22.0, 'longitude': 88.0},
    'Casablanca': {'latitude': 33.5, 'longitude': -7.5},
    'Curitiba': {'latitude': -25.5, 'longitude': -49.0},
    'Suape': {'latitude': -8.5, 'longitude': -35.0},
    'Santos': {'latitude': -24.0, 'longitude': -46.0},
    'Panama': {'latitude': 9.0, 'longitude': -79.5},
    'Niigata': {'latitude': 38.0, 'longitude': 139.0},
    'Sendai': {'latitude': 38.5, 'longitude': 141.0},
    'Yamaguchi': {'latitude': 34.0, 'longitude': 131.5},
    'Kobe': {'latitude': 34.5, 'longitude': 135.0},
    'Oita': {'latitude': 33.0, 'longitude': 131.5},
    'Aomori': {'latitude': 41.0, 'longitude': 140.5},
    'Seoul': {'latitude': 37.5, 'longitude': 127.0},
    'Busan': {'latitude': 35.0, 'longitude': 129.0},
    'Yeosu': {'latitude': 34.5, 'longitude': 127.5},
    'Harbin': {'latitude': 45.5, 'longitude': 126.5},
    'Changchun': {'latitude': 44.0, 'longitude': 125.5},
    'Shenyang': {'latitude': 42.0, 'longitude': 123.5},
    'Hohhot': {'latitude': 41.0, 'longitude': 111.5},
    'Urumqi': {'latitude': 44.0, 'longitude': 87.5},
    'Xining': {'latitude': 36.5, 'longitude': 101.5},
    'Lanzhou': {'latitude': 36.0, 'longitude': 104.0},
    'Beijing': {'latitude': 40.0, 'longitude': 116.5},
    'Tianjin': {'latitude': 39.5, 'longitude': 117.5},
    'Shijiazhuang': {'latitude': 38.0, 'longitude': 114.5},
    'Taiyuan': {'latitude': 38.0, 'longitude': 112.5},
    'Yinchuan': {'latitude': 38.5, 'longitude': 106.0},
    'Jinan': {'latitude': 36.5, 'longitude': 117.0},
    'Xian': {'latitude': 34.5, 'longitude': 109.0},
    'Zhengzhou': {'latitude': 35.0, 'longitude': 113.5},
    'Hefei': {'latitude': 32.0, 'longitude': 117.0},
    'Nanchang': {'latitude': 28.5, 'longitude': 116.0},
    'Wuhan': {'latitude': 30.5, 'longitude': 114.5},
    'Changsha': {'latitude': 28.0, 'longitude': 113.0},
    'Nanjing': {'latitude': 32.0, 'longitude': 119.0},
    'Hangzhou': {'latitude': 30.5, 'longitude': 120.0},
    'Fuzhou': {'latitude': 26.0, 'longitude': 119.5},
    'Guangzhou': {'latitude': 23.0, 'longitude': 113.5},
    'Nanning': {'latitude': 23.0, 'longitude': 108.5},
    'Haikou': {'latitude': 20.0, 'longitude': 110.0},
    'Chongqing': {'latitude': 29.5, 'longitude': 107.0},
    'Guiyang': {'latitude': 26.5, 'longitude': 106.5},
    'Kunming': {'latitude': 25.0, 'longitude': 102.5},
    'Chengdu': {'latitude': 30.5, 'longitude': 104.0},
    'Shanghai': {'latitude': 31.0, 'longitude': 121.5},
    'Kansas-City': {'latitude': 39.0, 'longitude': -94.5},
    'Oklahoma-City': {'latitude': 35.5, 'longitude': -97.5},
    'Columbia': {'latitude': 34.0, 'longitude': -81.0},
    'Tallahassee': {'latitude': 30.5, 'longitude': -84.5},
    'Raleigh': {'latitude': 35.5, 'longitude': -78.5},
}

# Population (for weighting)
POPULATION: dict[str, int] = {
    'Augsburg': 300000, 'Karlsruhe': 310000, 'Mannheim': 320000,
    'Bergerac': 27000, 'Le Mans': 143000,
    'Delhi': 19000000, 'Jamnagar': 600000, 'Haldia': 200000,
    'Casablanca': 3500000,
    'Curitiba': 1900000, 'Suape': 30000, 'Santos': 430000,
    'Panama': 880000,
    'Niigata': 800000, 'Sendai': 1000000, 'Yamaguchi': 145000,
    'Kobe': 1500000, 'Oita': 470000, 'Aomori': 280000,
    'Seoul': 9700000, 'Busan': 3400000, 'Yeosu': 300000,
    'Harbin': 30290000, 'Changchun': 23170000, 'Shenyang': 41550000,
    'Hohhot': 23800000, 'Urumqi': 26230000, 'Xining': 5930000,
    'Lanzhou': 24580000, 'Beijing': 21830000, 'Tianjin': 13640000,
    'Shijiazhuang': 78780000, 'Taiyuan': 34460000, 'Yinchuan': 7290000,
    'Jinan': 100800000, 'Zhengzhou': 97850000, 'Xian': 39520000,
    'Nanjing': 85260000, 'Hangzhou': 66269999, 'Hefei': 61270000,
    'Fuzhou': 41880000, 'Nanchang': 45280000, 'Wuhan': 58440000,
    'Changsha': 66040000, 'Guangzhou': 127060000, 'Nanning': 50470000,
    'Haikou': 10270000, 'Chongqing': 32130000, 'Guiyang': 38560000,
    'Kunming': 46930000, 'Chengdu': 83470000, 'Shanghai': 24800000,
    'Kansas-City': 510000, 'Oklahoma-City': 650000,
    'Columbia': 133000, 'Tallahassee': 194000, 'Raleigh': 480000,
}

# Region definitions
REGION_MAP: dict[str, list[str]] = {
    'Germany': ['Augsburg', 'Karlsruhe', 'Mannheim'],
    'France': ['Bergerac', 'Le Mans'],
    'India': ['Delhi', 'Jamnagar', 'Haldia'],
    'Morocco': ['Casablanca'],
    'Brazil': ['Curitiba', 'Suape', 'Santos'],
    'Panama': ['Panama'],
    'Japan': ['Niigata', 'Sendai', 'Yamaguchi', 'Kobe', 'Oita', 'Aomori'],
    'South Korea': ['Seoul', 'Busan', 'Yeosu'],
    'China North': ['Harbin', 'Changchun', 'Shenyang', 'Hohhot', 'Urumqi',
                    'Xining', 'Lanzhou', 'Beijing', 'Tianjin', 'Shijiazhuang',
                    'Taiyuan', 'Yinchuan', 'Jinan', 'Xian'],
    'China Central': ['Zhengzhou', 'Hefei', 'Nanchang', 'Wuhan', 'Changsha'],
    'China South': ['Nanjing', 'Hangzhou', 'Fuzhou', 'Guangzhou', 'Nanning',
                    'Haikou', 'Chongqing', 'Guiyang', 'Kunming', 'Chengdu',
                    'Shanghai'],
    'USA (Kansas & Oklahoma)': ['Kansas-City', 'Oklahoma-City'],
    'USA (Columbia)': ['Columbia'],
    'USA (Tallahassee)': ['Tallahassee'],
    'USA (Raleigh)': ['Raleigh'],
}

# Reverse lookup
CITY_TO_REGION: dict[str, str] = {
    city: region for region, cities in REGION_MAP.items() for city in cities
}

# Default regions to show
DEFAULT_REGIONS = ['China North', 'China South', 'China Central', 'Japan',
                   'South Korea', 'India']
