"""Data access layer — Coal Desk CDD Dashboard (v2).

All queries go to dna_snbx_weather.coal_desk (sandbox tables).
Uses SP token fallback if user token has scope issues.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from _config import (
    CITY_LOCATIONS, POPULATION, REGION_MAP, CITY_TO_REGION,
    BASE_TEMP, SEASON_START_MONTH, SEASON_START_DAY, HIST_START_YEAR, HIST_END_YEAR,
    TABLE_HIST, TABLE_FCST, CURVE_HIST, CURVE_FCST, MODEL_HIST, MODEL_FCST,
    TABLE_PRECIP_HIST, TABLE_PRECIP_FCST, CURVE_PRECIP_HIST, CURVE_PRECIP_FCST,
)

# ─── Config ──────────────────────────────────────────────────────────────────────
_raw_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_HOST = _raw_host if _raw_host.startswith("https://") else f"https://{_raw_host}"
WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "").split("/")[-1]
COAL_DESK_SCHEMA = "dna_snbx_weather.coal_desk"


# ─── Query execution (with SP fallback) ─────────────────────────────────────────

def run_query(query: str) -> pd.DataFrame:
    """Execute SQL via Statement API. Falls back to SP token on 403."""
    user_token = None
    try:
        user_token = st.context.headers.get("x-forwarded-access-token")
    except Exception:
        pass

    def _get_sp_headers():
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        h = {"Content-Type": "application/json"}
        # Try different SDK auth patterns
        try:
            tok = w.config.authenticate()
            if isinstance(tok, dict):
                h.update(tok)
            elif isinstance(tok, str):
                h["Authorization"] = f"Bearer {tok}"
        except TypeError:
            pass
        # Fallback: access token directly from config
        if "Authorization" not in h:
            try:
                h["Authorization"] = f"Bearer {w.config.token}"
            except Exception:
                try:
                    h["Authorization"] = f"Bearer {w.config.host_credentials_provider()().get('Authorization', '')}"
                except Exception:
                    pass
        return h

    if user_token:
        headers = {"Authorization": f"Bearer {user_token}", "Content-Type": "application/json"}
    else:
        headers = _get_sp_headers()

    resp = requests.post(
        f"{DATABRICKS_HOST}/api/2.0/sql/statements/",
        headers=headers,
        json={"warehouse_id": WAREHOUSE_ID, "statement": query, "wait_timeout": "50s"},
        timeout=60,
    )

    # Fallback: retry with SP token if user token fails
    if resp.status_code == 403 and user_token:
        sp_headers = _get_sp_headers()
        resp = requests.post(
            f"{DATABRICKS_HOST}/api/2.0/sql/statements/",
            headers=sp_headers,
            json={"warehouse_id": WAREHOUSE_ID, "statement": query, "wait_timeout": "50s"},
            timeout=60,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    status = data.get("status", {})
    if status.get("state") == "FAILED":
        raise RuntimeError(f"Query failed: {status.get('error', {}).get('message', 'Unknown')}")

    columns = [c["name"] for c in data.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = data.get("result", {}).get("data_array", [])
    return pd.DataFrame(rows, columns=columns)


# ─── Data loaders (sandbox tables only) ────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_historical(region: str) -> pd.DataFrame:
    query = f"SELECT date, temperature, cdd FROM {COAL_DESK_SCHEMA}.coal_desk_cdd_historical WHERE region = '{region}' ORDER BY date"
    df = run_query(query)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df['temperature'] = df['temperature'].astype(float)
    df['cdd'] = df['cdd'].astype(float)
    df['city'] = region
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(region: str) -> pd.DataFrame:
    query = f"SELECT date, temperature, cdd FROM {COAL_DESK_SCHEMA}.coal_desk_cdd WHERE region = '{region}' ORDER BY date"
    df = run_query(query)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df['temperature'] = df['temperature'].astype(float)
    df['cdd'] = df['cdd'].astype(float)
    df['city'] = region
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_city_timeseries(city: str, lookback_days: int = 90) -> pd.DataFrame:
    query = f"SELECT date, value as temperature FROM {COAL_DESK_SCHEMA}.coal_desk_forecasts WHERE city = '{city}' AND parameter = 't_mean_2m_24h' ORDER BY date"
    df = run_query(query)
    if df.empty:
        return pd.DataFrame(columns=['date', 'temperature'])
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df['temperature'] = df['temperature'].astype(float)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_anomalies(target_date) -> pd.DataFrame:
    """Compute forecast anomaly vs climatology for all cities.
    Tries sandbox first, falls back to production tables."""
    target_str = str(target_date)
    doy = pd.Timestamp(target_date).day_of_year

    # Try sandbox forecasts first
    fcst_q = f"SELECT city, AVG(value) as temperature FROM {COAL_DESK_SCHEMA}.coal_desk_forecasts WHERE parameter = 't_mean_2m_24h' AND CAST(date AS DATE) = '{target_str}' GROUP BY city"
    fcst_df = run_query(fcst_q)

    # Fallback to production forecast table if sandbox empty for this date
    if fcst_df.empty:
        all_cities = list(CITY_LOCATIONS.keys())
        coord_parts = []
        for city in all_cities:
            loc = CITY_LOCATIONS[city]
            coord_parts.append(f"(latitude = {loc['latitude']} AND longitude = {loc['longitude']})")
        coord_filter = " OR ".join(coord_parts)
        fcst_q = f"""
        SELECT CAST(delivery_start AS DATE) as date, AVG(value) as temperature,
               latitude, longitude
        FROM {TABLE_FCST}
        WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_FCST}'
          AND CAST(delivery_start AS DATE) = '{target_str}'
          AND ({coord_filter})
        GROUP BY CAST(delivery_start AS DATE), latitude, longitude
        """
        fcst_df = run_query(fcst_q)
        if fcst_df.empty:
            return pd.DataFrame()
        fcst_df['temperature'] = fcst_df['temperature'].astype(float)
        fcst_df['latitude'] = fcst_df['latitude'].astype(float)
        fcst_df['longitude'] = fcst_df['longitude'].astype(float)
        coord_to_city = {}
        for city in all_cities:
            loc = CITY_LOCATIONS[city]
            coord_to_city[(loc['latitude'], loc['longitude'])] = city
        fcst_df['city'] = fcst_df.apply(
            lambda r: coord_to_city.get((r['latitude'], r['longitude']), '?'), axis=1
        )
        fcst_df = fcst_df[fcst_df['city'] != '?']
    else:
        fcst_df['temperature'] = fcst_df['temperature'].astype(float)

    fcst_df['region'] = fcst_df['city'].map(CITY_TO_REGION)

    # Get city-level climatology from production ERA5
    all_cities = list(CITY_LOCATIONS.keys())
    coord_parts = []
    for city in all_cities:
        loc = CITY_LOCATIONS[city]
        coord_parts.append(f"(latitude = {loc['latitude']} AND longitude = {loc['longitude']})")
    coord_filter = " OR ".join(coord_parts)
    clim_q = f"""
    SELECT AVG(value) as climatology, latitude, longitude
    FROM {TABLE_HIST}
    WHERE model = '{MODEL_HIST}' AND curve_name = '{CURVE_HIST}'
      AND DAYOFYEAR(delivery_start) = {doy}
      AND YEAR(delivery_start) BETWEEN {HIST_START_YEAR} AND {HIST_END_YEAR}
      AND ({coord_filter})
    GROUP BY latitude, longitude
    """
    clim_df = run_query(clim_q)

    if not clim_df.empty:
        clim_df['climatology'] = clim_df['climatology'].astype(float)
        clim_df['latitude'] = clim_df['latitude'].astype(float)
        clim_df['longitude'] = clim_df['longitude'].astype(float)
        coord_to_city = {}
        for city in all_cities:
            loc = CITY_LOCATIONS[city]
            coord_to_city[(loc['latitude'], loc['longitude'])] = city
        clim_df['city'] = clim_df.apply(
            lambda r: coord_to_city.get((r['latitude'], r['longitude']), '?'), axis=1
        )
        merged = fcst_df.merge(clim_df[['city', 'climatology']], on='city', how='left')
        merged['anomaly'] = merged['temperature'] - merged['climatology'].fillna(merged['temperature'])
    else:
        merged = fcst_df.copy()
        merged['climatology'] = np.nan
        merged['anomaly'] = 0.0

    merged['latitude'] = merged['city'].map(lambda c: CITY_LOCATIONS[c]['latitude'] if c in CITY_LOCATIONS else 0)
    merged['longitude'] = merged['city'].map(lambda c: CITY_LOCATIONS[c]['longitude'] if c in CITY_LOCATIONS else 0)
    return merged


# ─── Gridded anomaly map data ────────────────────────────────────────────────────

MAP_REGIONS = {
    'East Asia': {'lat_min': 20, 'lat_max': 46.5, 'lon_min': 90, 'lon_max': 146.5},
    'Europe': {'lat_min': 36, 'lat_max': 72.5, 'lon_min': -13, 'lon_max': 36.5},
    'US': {'lat_min': 28.5, 'lat_max': 55.5, 'lon_min': -130, 'lon_max': -70},
}


@st.cache_data(ttl=1800, show_spinner=False)
def load_gridded_precip_deviation(map_region: str, start_date, end_date) -> pd.DataFrame:
    """Load gridded precipitation deviation (forecast vs climatology) for a date range."""
    bounds = MAP_REGIONS[map_region]
    start_str, end_str = str(start_date), str(end_date)
    fcst_q = f"""
    SELECT AVG(value) as precipitation, latitude, longitude
    FROM {TABLE_PRECIP_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_PRECIP_FCST}'
      AND CAST(delivery_start AS DATE) BETWEEN '{start_str}' AND '{end_str}'
      AND latitude BETWEEN {bounds['lat_min']} AND {bounds['lat_max']}
      AND longitude BETWEEN {bounds['lon_min']} AND {bounds['lon_max']}
      AND MOD(CAST(latitude * 2 AS INT), 2) = 0
      AND MOD(CAST(longitude * 2 AS INT), 2) = 0
    GROUP BY latitude, longitude
    """
    fcst_df = run_query(fcst_q)
    if fcst_df.empty:
        return pd.DataFrame()
    fcst_df['precipitation'] = fcst_df['precipitation'].astype(float)
    fcst_df['latitude'] = fcst_df['latitude'].astype(float)
    fcst_df['longitude'] = fcst_df['longitude'].astype(float)
    doy_start = pd.Timestamp(start_date).day_of_year
    doy_end = pd.Timestamp(end_date).day_of_year
    clim_q = f"""
    SELECT AVG(value) as climatology, latitude, longitude
    FROM {TABLE_PRECIP_HIST}
    WHERE model = '{MODEL_HIST}' AND curve_name = '{CURVE_PRECIP_HIST}'
      AND DAYOFYEAR(delivery_start) BETWEEN {doy_start} AND {doy_end}
      AND YEAR(delivery_start) BETWEEN {HIST_START_YEAR} AND {HIST_END_YEAR}
      AND latitude BETWEEN {bounds['lat_min']} AND {bounds['lat_max']}
      AND longitude BETWEEN {bounds['lon_min']} AND {bounds['lon_max']}
      AND MOD(CAST(latitude * 2 AS INT), 2) = 0
      AND MOD(CAST(longitude * 2 AS INT), 2) = 0
    GROUP BY latitude, longitude
    """
    clim_df = run_query(clim_q)
    if clim_df.empty:
        return fcst_df.assign(anomaly=0.0)
    clim_df['climatology'] = clim_df['climatology'].astype(float)
    clim_df['latitude'] = clim_df['latitude'].astype(float)
    clim_df['longitude'] = clim_df['longitude'].astype(float)
    merged = fcst_df.merge(clim_df, on=['latitude', 'longitude'], how='inner')
    merged['anomaly'] = merged['precipitation'] - merged['climatology']
    return merged


@st.cache_data(ttl=1800, show_spinner=False)
def load_gridded_anomalies_multiday(map_region: str, start_date, end_date) -> pd.DataFrame:
    """Load gridded forecast + climatology for a date range, at 1-deg resolution.
    Returns DataFrame with lat, lon, anomaly (averaged over the period)."""
    bounds = MAP_REGIONS[map_region]
    start_str = str(start_date)
    end_str = str(end_date)

    # Average forecast over the period
    fcst_q = f"""
    SELECT AVG(value) as temperature, latitude, longitude
    FROM {TABLE_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_FCST}'
      AND CAST(delivery_start AS DATE) BETWEEN '{start_str}' AND '{end_str}'
      AND latitude BETWEEN {bounds['lat_min']} AND {bounds['lat_max']}
      AND longitude BETWEEN {bounds['lon_min']} AND {bounds['lon_max']}
      AND MOD(CAST(latitude * 2 AS INT), 2) = 0
      AND MOD(CAST(longitude * 2 AS INT), 2) = 0
    GROUP BY latitude, longitude
    """
    fcst_df = run_query(fcst_q)
    if fcst_df.empty:
        return pd.DataFrame()

    fcst_df['temperature'] = fcst_df['temperature'].astype(float)
    fcst_df['latitude'] = fcst_df['latitude'].astype(float)
    fcst_df['longitude'] = fcst_df['longitude'].astype(float)

    # Average climatology over same day-of-year range
    doy_start = pd.Timestamp(start_date).day_of_year
    doy_end = pd.Timestamp(end_date).day_of_year
    clim_q = f"""
    SELECT AVG(value) as climatology, latitude, longitude
    FROM {TABLE_HIST}
    WHERE model = '{MODEL_HIST}' AND curve_name = '{CURVE_HIST}'
      AND DAYOFYEAR(delivery_start) BETWEEN {doy_start} AND {doy_end}
      AND YEAR(delivery_start) BETWEEN {HIST_START_YEAR} AND {HIST_END_YEAR}
      AND latitude BETWEEN {bounds['lat_min']} AND {bounds['lat_max']}
      AND longitude BETWEEN {bounds['lon_min']} AND {bounds['lon_max']}
      AND MOD(CAST(latitude * 2 AS INT), 2) = 0
      AND MOD(CAST(longitude * 2 AS INT), 2) = 0
    GROUP BY latitude, longitude
    """
    clim_df = run_query(clim_q)
    if clim_df.empty:
        return fcst_df.assign(anomaly=0.0)

    clim_df['climatology'] = clim_df['climatology'].astype(float)
    clim_df['latitude'] = clim_df['latitude'].astype(float)
    clim_df['longitude'] = clim_df['longitude'].astype(float)

    merged = fcst_df.merge(clim_df, on=['latitude', 'longitude'], how='inner')
    merged['anomaly'] = merged['temperature'] - merged['climatology']
    return merged


@st.cache_data(ttl=1800, show_spinner=False)
def load_gridded_anomalies(map_region: str, target_date) -> pd.DataFrame:
    """Load gridded forecast + climatology and compute anomalies for a map region.
    Queries production tables at 2-degree resolution for performance."""
    bounds = MAP_REGIONS[map_region]
    target_str = str(target_date)
    doy = pd.Timestamp(target_date).day_of_year

    # Forecast for target date (subsample to 2-degree grid)
    fcst_q = f"""
    SELECT AVG(value) as temperature, latitude, longitude
    FROM {TABLE_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_FCST}'
      AND CAST(delivery_start AS DATE) = '{target_str}'
      AND latitude BETWEEN {bounds['lat_min']} AND {bounds['lat_max']}
      AND longitude BETWEEN {bounds['lon_min']} AND {bounds['lon_max']}
      AND MOD(CAST(latitude * 2 AS INT), 4) = 0
      AND MOD(CAST(longitude * 2 AS INT), 4) = 0
    GROUP BY latitude, longitude
    """
    fcst_df = run_query(fcst_q)
    if fcst_df.empty:
        return pd.DataFrame()

    fcst_df['temperature'] = fcst_df['temperature'].astype(float)
    fcst_df['latitude'] = fcst_df['latitude'].astype(float)
    fcst_df['longitude'] = fcst_df['longitude'].astype(float)

    # Climatology (ERA5 day-of-year average 2000-2024, same grid)
    clim_q = f"""
    SELECT AVG(value) as climatology, latitude, longitude
    FROM {TABLE_HIST}
    WHERE model = '{MODEL_HIST}' AND curve_name = '{CURVE_HIST}'
      AND DAYOFYEAR(delivery_start) = {doy}
      AND YEAR(delivery_start) BETWEEN {HIST_START_YEAR} AND {HIST_END_YEAR}
      AND latitude BETWEEN {bounds['lat_min']} AND {bounds['lat_max']}
      AND longitude BETWEEN {bounds['lon_min']} AND {bounds['lon_max']}
      AND MOD(CAST(latitude * 2 AS INT), 4) = 0
      AND MOD(CAST(longitude * 2 AS INT), 4) = 0
    GROUP BY latitude, longitude
    """
    clim_df = run_query(clim_q)
    if clim_df.empty:
        return fcst_df.assign(anomaly=0.0)

    clim_df['climatology'] = clim_df['climatology'].astype(float)
    clim_df['latitude'] = clim_df['latitude'].astype(float)
    clim_df['longitude'] = clim_df['longitude'].astype(float)

    merged = fcst_df.merge(clim_df, on=['latitude', 'longitude'], how='inner')
    merged['anomaly'] = merged['temperature'] - merged['climatology']
    return merged


@st.cache_data(ttl=1800, show_spinner=False)
def load_precomputed_cdd() -> pd.DataFrame:
    query = f"SELECT date, region, model, cdd, temperature, n_cities FROM {COAL_DESK_SCHEMA}.coal_desk_cdd ORDER BY date"
    df = run_query(query)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        df['cdd'] = df['cdd'].astype(float)
        df['temperature'] = df['temperature'].astype(float)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_precomputed_historical() -> pd.DataFrame:
    query = f"SELECT date, region, cdd, temperature FROM {COAL_DESK_SCHEMA}.coal_desk_cdd_historical ORDER BY date"
    df = run_query(query)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        df['cdd'] = df['cdd'].astype(float)
        df['temperature'] = df['temperature'].astype(float)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_precomputed_forecasts(parameter: str = None, model: str = None) -> pd.DataFrame:
    conditions = []
    if parameter:
        conditions.append(f"parameter = '{parameter}'")
    if model:
        conditions.append(f"model = '{model}'")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"SELECT date, city, value, parameter, model, label FROM {COAL_DESK_SCHEMA}.coal_desk_forecasts {where} ORDER BY date"
    df = run_query(query)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        df['value'] = df['value'].astype(float)
    return df


# ─── Current year CDD from production tables ─────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def load_current_year_cdd(region: str) -> pd.DataFrame:
    """Load current year CDD from ERA5 actuals + ECMWF-ENS forecast gap fill."""
    current_year = datetime.now().year
    season_start = f"{current_year}-{SEASON_START_MONTH:02d}-{SEASON_START_DAY:02d}"
    cities = REGION_MAP[region]
    coord_parts = []
    for city in cities:
        loc = CITY_LOCATIONS[city]
        coord_parts.append(f"(latitude = {loc['latitude']} AND longitude = {loc['longitude']})")
    coord_filter = " OR ".join(coord_parts)

    era5_q = f"""
    SELECT CAST(delivery_start AS DATE) as date, value as temperature,
           latitude, longitude
    FROM {TABLE_HIST}
    WHERE model = '{MODEL_HIST}' AND curve_name = '{CURVE_HIST}'
      AND delivery_start >= '{season_start}'
      AND ({coord_filter})
    ORDER BY delivery_start
    """
    era5_df = run_query(era5_q)

    gap_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    fcst_q = f"""
    SELECT CAST(delivery_start AS DATE) as date, AVG(value) as temperature,
           latitude, longitude
    FROM {TABLE_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_FCST}'
      AND CAST(delivery_start AS DATE) >= '{gap_start}'
      AND ({coord_filter})
    GROUP BY CAST(delivery_start AS DATE), latitude, longitude
    ORDER BY date
    """
    fcst_df = run_query(fcst_q)

    coord_to_city = {}
    for city in cities:
        loc = CITY_LOCATIONS[city]
        coord_to_city[(loc['latitude'], loc['longitude'])] = city

    frames = []
    for df in (era5_df, fcst_df):
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['temperature'] = df['temperature'].astype(float)
            df['latitude'] = df['latitude'].astype(float)
            df['longitude'] = df['longitude'].astype(float)
            df['city'] = df.apply(
                lambda r: coord_to_city.get((r['latitude'], r['longitude']), '?'), axis=1
            )
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=['date', 'cdd'])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(['date', 'city']).drop_duplicates(
        subset=['date', 'city'], keep='first'
    )
    combined['cdd_city'] = (combined['temperature'] - BASE_TEMP).clip(lower=0)
    pivot = combined.pivot_table(index='date', columns='city', values='cdd_city', aggfunc='mean')
    available = [c for c in cities if c in pivot.columns]
    if not available:
        return pd.DataFrame(columns=['date', 'cdd'])

    pops = np.array([POPULATION[c] for c in available], dtype=float)
    weights = pops / pops.sum()
    weighted = (pivot[available].values * weights[None, :]).sum(axis=1)
    return pd.DataFrame({'date': pivot.index, 'cdd': weighted}).sort_values('date').reset_index(drop=True)


# ─── CDD computation ────────────────────────────────────────────────────────────

def compute_region_cdd(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['date', 'cdd'])
    if 'cdd' in df.columns:
        return df[['date', 'cdd']].copy()
    df = df.copy()
    df['cdd'] = (df['temperature'] - BASE_TEMP).clip(lower=0)
    return df[['date', 'cdd']].copy()


def compute_cumulative(cdd_df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    end = pd.Timestamp(year=year + 1, month=SEASON_START_MONTH, day=SEASON_START_DAY - 1)
    mask = (cdd_df['date'] >= start) & (cdd_df['date'] <= end)
    s = cdd_df[mask].copy().sort_values('date')
    if s.empty:
        return pd.DataFrame(columns=['date', 'day_of_season', 'cumulative_cdd'])
    s['cumulative_cdd'] = s['cdd'].cumsum()
    s['day_of_season'] = range(1, len(s) + 1)
    return s[['date', 'day_of_season', 'cumulative_cdd']]


def compute_normal(cdd_df: pd.DataFrame) -> pd.DataFrame:
    curves = []
    for yr in range(HIST_START_YEAR, HIST_END_YEAR + 1):
        c = compute_cumulative(cdd_df, yr)
        if not c.empty:
            c['year'] = yr
            curves.append(c[['day_of_season', 'cumulative_cdd', 'year']])
    if not curves:
        return pd.DataFrame(columns=['day_of_season', 'mean', 'std', 'upper', 'lower'])
    combined = pd.concat(curves, ignore_index=True)
    stats = combined.groupby('day_of_season')['cumulative_cdd'].agg(['mean', 'std']).reset_index()
    stats['upper'] = stats['mean'] + stats['std']
    stats['lower'] = (stats['mean'] - stats['std']).clip(lower=0)
    return stats
