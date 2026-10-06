"""Configuration — Coal Desk Weather / CDD Report.

Cities, regions, populations, constants.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

# Degree-day parameters
BASE_TEMP = 18.0  # Celsius — base for both CDD and HDD
# Legacy fixed season start, still used by the older app.py / _data.py
SEASON_START_MONTH = 4
SEASON_START_DAY = 15
# Seasonal switch (app_v2): regions with a clear seasonality show cumulative CDD from
# 1 May and cumulative HDD from 1 Oct. Other regions stay on CDD (12 months from 1 May).
CDD_SEASON_START = (5, 1)    # (month, day)
HDD_SEASON_START = (10, 1)
HIST_START_YEAR = 2000
HIST_END_YEAR = 2024
# Trailing window for the "past 5-year average" comparison (most recent 5 hist years)
FIVE_YEAR_START = HIST_END_YEAR - 4   # 2020–2024 inclusive

# Source tables — Temperature (sandbox, no prod dependency)
TABLE_HIST = "dna_snbx_weather.coal_desk.temperature_actuals"
TABLE_FCST = "dna_snbx_weather.coal_desk.temperature_forecast"
# Extended-range gridded forecast (ECMWF-vareps, 44d) — populated by the refresh
# pipeline. Used for the week 3-6 anomaly maps. Falls back to "No data" if absent.
TABLE_FCST_VAREPS = "dna_snbx_weather.coal_desk.temperature_forecast_vareps"
TEMP_CLIM = "dna_snbx_weather.coal_desk.temperature_climatology"
CURVE_HIST = "t_mean_2m_24h_c_ecmwf_era5_p1d"
CURVE_FCST = "t_mean_2m_24h_c_ecmwf_ens_p1d"
CURVE_FCST_VAREPS = "t_mean_2m_24h_c_ecmwf_vareps_p1d"
MODEL_HIST = "ecmwf-era5"
MODEL_FCST = "ecmwf-ens"
MODEL_FCST_VAREPS = "ecmwf-vareps"

# Source tables — Precipitation (sandbox, no prod dependency)
TABLE_PRECIP_HIST = "dna_snbx_weather.coal_desk.precipitation_actuals"
TABLE_PRECIP_FCST = "dna_snbx_weather.coal_desk.precipitation_forecast"
PRECIP_CLIM = "dna_snbx_weather.coal_desk.precipitation_climatology"
CURVE_PRECIP_HIST = "precip_24h_mm_ecmwf_era5_p1d"
CURVE_PRECIP_FCST = "precip_24h_mm_ecmwf_ens_p1d"
# Precipitation climatology uses 'mix' model (1978–2025 long record)
MODEL_PRECIP_CLIM = "mix"
CURVE_PRECIP_CLIM = "precip_24h_mm_mix_p1d"

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

# Regions with a clear heating/cooling seasonality (the rest are tropical/sub-tropical
# and stay on CDD all year)
SEASONAL_REGIONS = {
    'Germany', 'France', 'Japan', 'South Korea',
    'China North', 'China Central', 'China South',
    'USA (Kansas & Oklahoma)', 'USA (Columbia)', 'USA (Tallahassee)', 'USA (Raleigh)',
}


def region_mode(region: str, today=None) -> str:
    """'cdd' or 'hdd' for the region on `today`: HDD from 1 Oct to 30 Apr, CDD from 1 May."""
    if region not in SEASONAL_REGIONS:
        return 'cdd'
    today = today or date.today()
    return 'hdd' if (today.month, today.day) >= HDD_SEASON_START or \
        (today.month, today.day) < CDD_SEASON_START else 'cdd'


def current_season_year(region: str, today=None) -> int:
    """Calendar year in which the region's current season started."""
    today = today or date.today()
    start = HDD_SEASON_START if region_mode(region, today) == 'hdd' else CDD_SEASON_START
    return today.year if (today.month, today.day) >= start else today.year - 1


def season_bounds(region: str, year: int, mode: str | None = None) -> tuple:
    """(start, end) Timestamps of the season that starts in `year` for the region."""
    mode = mode or region_mode(region)
    if mode == 'hdd':
        start = pd.Timestamp(year=year, month=HDD_SEASON_START[0], day=HDD_SEASON_START[1])
        end = pd.Timestamp(year=year + 1, month=CDD_SEASON_START[0], day=CDD_SEASON_START[1]) \
            - pd.Timedelta(days=1)
    else:
        start = pd.Timestamp(year=year, month=CDD_SEASON_START[0], day=CDD_SEASON_START[1])
        if region in SEASONAL_REGIONS:
            end = pd.Timestamp(year=year, month=HDD_SEASON_START[0], day=HDD_SEASON_START[1]) \
                - pd.Timedelta(days=1)
        else:
            end = pd.Timestamp(year=year + 1, month=CDD_SEASON_START[0], day=CDD_SEASON_START[1]) \
                - pd.Timedelta(days=1)
    return start, end


def season_label(region: str, year: int) -> str:
    """'2026' for a CDD season, '2026/27' for an HDD (winter) season."""
    return f"{year}/{str(year + 1)[-2:]}" if region_mode(region) == 'hdd' else str(year)
