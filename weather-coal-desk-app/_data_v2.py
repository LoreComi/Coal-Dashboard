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
    MODEL_PRECIP_CLIM, CURVE_PRECIP_CLIM,
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
    query = f"SELECT date, model, temperature, cdd FROM {COAL_DESK_SCHEMA}.coal_desk_cdd WHERE region = '{region}' ORDER BY model, date"
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
    WHERE model = '{MODEL_PRECIP_CLIM}' AND curve_name = '{CURVE_PRECIP_CLIM}'
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
def load_current_year_cdd_bulk(regions: tuple) -> pd.DataFrame:
    """Load current-year ERA5 actuals + ENS forecast for multiple regions in two queries.

    ERA5 actuals cover the whole season so far; ENS fills the 7-day gap and future dates.
    Returns DataFrame with columns: date, cdd, region.
    """
    current_year = datetime.now().year
    season_start = f"{current_year}-{SEASON_START_MONTH:02d}-{SEASON_START_DAY:02d}"
    gap_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    all_cities = [c for r in regions for c in REGION_MAP.get(r, [])]
    if not all_cities:
        return pd.DataFrame(columns=['date', 'cdd', 'region'])

    coord_parts = [
        f"(latitude = {CITY_LOCATIONS[c]['latitude']} AND longitude = {CITY_LOCATIONS[c]['longitude']})"
        for c in all_cities
    ]
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
    ens_q = f"""
    SELECT CAST(delivery_start AS DATE) as date, AVG(value) as temperature,
           latitude, longitude
    FROM {TABLE_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_FCST}'
      AND CAST(delivery_start AS DATE) >= '{gap_start}'
      AND ({coord_filter})
    GROUP BY CAST(delivery_start AS DATE), latitude, longitude
    ORDER BY date
    """

    era5_df = run_query(era5_q)
    ens_df = run_query(ens_q)

    coord_to_city = {
        (CITY_LOCATIONS[c]['latitude'], CITY_LOCATIONS[c]['longitude']): c
        for c in all_cities
    }

    frames = []
    for df in (era5_df, ens_df):
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['temperature'] = df['temperature'].astype(float)
            df['latitude'] = df['latitude'].astype(float)
            df['longitude'] = df['longitude'].astype(float)
            df['city'] = df.apply(lambda r: coord_to_city.get((r['latitude'], r['longitude']), '?'), axis=1)
            df['region'] = df['city'].map(CITY_TO_REGION)
            frames.append(df[df['city'] != '?'])

    if not frames:
        return pd.DataFrame(columns=['date', 'cdd', 'region'])

    # ERA5 rows appear first; pandas stable sort keeps ERA5 priority on dedup
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(['date', 'city']).drop_duplicates(subset=['date', 'city'], keep='first')

    region_results = []
    for r in regions:
        cities = REGION_MAP.get(r, [])
        if not cities:
            continue
        rd = combined[combined['region'] == r].copy()
        if rd.empty:
            continue
        rd['cdd_city'] = (rd['temperature'] - BASE_TEMP).clip(lower=0)
        pivot = rd.pivot_table(index='date', columns='city', values='cdd_city', aggfunc='mean')
        avail = [c for c in cities if c in pivot.columns]
        if not avail:
            continue
        w = np.array([POPULATION[c] for c in avail], dtype=float)
        w = w / w.sum()
        weighted = (pivot[avail].values * w[None, :]).sum(axis=1)
        region_results.append(pd.DataFrame({'date': pivot.index, 'cdd': weighted, 'region': r}))

    if not region_results:
        return pd.DataFrame(columns=['date', 'cdd', 'region'])

    return pd.concat(region_results, ignore_index=True).sort_values(['region', 'date']).reset_index(drop=True)


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


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast_spread_bulk(regions: tuple) -> pd.DataFrame:
    """Load ensemble temperature 25th/75th percentile for all cities in the given regions (one query).

    Returns a DataFrame with columns: date, temp_p25, temp_p75, city, region.
    Used to build the ensemble uncertainty band on the cumulative CDD chart.
    """
    all_cities = [c for r in regions for c in REGION_MAP.get(r, [])]
    if not all_cities:
        return pd.DataFrame()

    coord_parts = []
    for city in all_cities:
        loc = CITY_LOCATIONS[city]
        coord_parts.append(f"(latitude = {loc['latitude']} AND longitude = {loc['longitude']})")
    coord_filter = " OR ".join(coord_parts)

    query = f"""
    SELECT CAST(delivery_start AS DATE) as date,
           PERCENTILE(value, 0.25) as temp_p25,
           PERCENTILE(value, 0.75) as temp_p75,
           latitude, longitude
    FROM {TABLE_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_FCST}'
      AND delivery_start >= CURRENT_DATE()
      AND ({coord_filter})
    GROUP BY CAST(delivery_start AS DATE), latitude, longitude
    ORDER BY date
    """
    df = run_query(query)
    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    for col in ('temp_p25', 'temp_p75', 'latitude', 'longitude'):
        df[col] = df[col].astype(float)

    coord_to_city = {
        (CITY_LOCATIONS[c]['latitude'], CITY_LOCATIONS[c]['longitude']): c
        for c in all_cities
    }
    df['city'] = df.apply(lambda r: coord_to_city.get((r['latitude'], r['longitude']), '?'), axis=1)
    df['region'] = df['city'].map(CITY_TO_REGION)
    return df


def compute_ensemble_spread(spread_bulk_df: pd.DataFrame, region: str, cum_current: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative CDD lower/upper bounds from ensemble 25th/75th percentile.

    Returns DataFrame with day_of_season, cumulative_lower, cumulative_upper.
    The envelope starts from the last actual (observed) cumulative value.
    """
    if spread_bulk_df.empty or cum_current.empty:
        return pd.DataFrame(columns=['day_of_season', 'cumulative_lower', 'cumulative_upper'])

    cities = REGION_MAP.get(region, [])
    region_spread = spread_bulk_df[spread_bulk_df['region'] == region].copy()
    if region_spread.empty or not cities:
        return pd.DataFrame(columns=['day_of_season', 'cumulative_lower', 'cumulative_upper'])

    rows = []
    for date, grp in region_spread.groupby('date'):
        grp = grp.set_index('city')
        avail = [c for c in cities if c in grp.index]
        if not avail:
            continue
        w = np.array([POPULATION[c] for c in avail], dtype=float)
        w = w / w.sum()
        cdd_min = (np.maximum(grp.loc[avail, 'temp_p25'].values - BASE_TEMP, 0) * w).sum()
        cdd_max = (np.maximum(grp.loc[avail, 'temp_p75'].values - BASE_TEMP, 0) * w).sum()
        rows.append({'date': date, 'cdd_min': cdd_min, 'cdd_max': cdd_max})

    if not rows:
        return pd.DataFrame(columns=['day_of_season', 'cumulative_lower', 'cumulative_upper'])

    spread_cdd = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

    today = pd.Timestamp.today().normalize()
    actual_part = cum_current[cum_current['date'] <= today]
    if actual_part.empty:
        return pd.DataFrame(columns=['day_of_season', 'cumulative_lower', 'cumulative_upper'])

    baseline = float(actual_part['cumulative_cdd'].iloc[-1])
    last_day = int(actual_part['day_of_season'].iloc[-1])
    last_date = actual_part['date'].iloc[-1]

    spread_future = spread_cdd[spread_cdd['date'] > last_date].copy().reset_index(drop=True)
    if spread_future.empty:
        return pd.DataFrame(columns=['day_of_season', 'cumulative_lower', 'cumulative_upper'])

    spread_future['cumulative_lower'] = baseline + spread_future['cdd_min'].cumsum()
    spread_future['cumulative_upper'] = baseline + spread_future['cdd_max'].cumsum()
    spread_future['day_of_season'] = range(last_day + 1, last_day + 1 + len(spread_future))

    return spread_future[['day_of_season', 'cumulative_lower', 'cumulative_upper']]


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


def load_all_historical_cumulative(historical_cdd: pd.DataFrame) -> dict:
    """Pre-compute cumulative CDD DataFrames for every historical year (2000–2024)."""
    result = {}
    for year in range(HIST_START_YEAR, HIST_END_YEAR + 1):
        cum = compute_cumulative(historical_cdd, year)
        if not cum.empty:
            result[year] = cum
    return result


def compute_daily_cdd_climatology_v2(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-year mean and std of daily CDD from pre-aggregated historical data (2000–2024)."""
    if hist_df.empty or 'cdd' not in hist_df.columns:
        return pd.DataFrame(columns=['day_of_year', 'mean_cdd', 'std_cdd'])
    df = hist_df[['date', 'cdd']].copy()
    df['day_of_year'] = pd.to_datetime(df['date']).dt.day_of_year
    df['year'] = pd.to_datetime(df['date']).dt.year
    df = df[df['year'].between(HIST_START_YEAR, HIST_END_YEAR)]
    stats = df.groupby('day_of_year')['cdd'].agg(['mean', 'std']).reset_index()
    stats.columns = ['day_of_year', 'mean_cdd', 'std_cdd']
    stats['std_cdd'] = stats['std_cdd'].fillna(0)
    return stats


def compute_temperature_climatology_simple(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-year mean and std of temperature from pre-aggregated historical data (2000–2024)."""
    if hist_df.empty or 'temperature' not in hist_df.columns:
        return pd.DataFrame(columns=['day_of_year', 'mean_temp', 'std_temp'])
    df = hist_df[['date', 'temperature']].copy()
    df['day_of_year'] = pd.to_datetime(df['date']).dt.day_of_year
    df['year'] = pd.to_datetime(df['date']).dt.year
    df = df[df['year'].between(HIST_START_YEAR, HIST_END_YEAR)]
    stats = df.groupby('day_of_year')['temperature'].agg(['mean', 'std']).reset_index()
    stats.columns = ['day_of_year', 'mean_temp', 'std_temp']
    stats['std_temp'] = stats['std_temp'].fillna(0)
    return stats


def compute_similar_years(historical_cdd: pd.DataFrame, current_cum: pd.DataFrame, n_similar: int = 5) -> list:
    """Return the n historical years whose CDD trajectory best matches the current year's.

    Similarity = weighted blend of RMSE (70%) and rate-of-change RMSE (30%) over
    the days elapsed so far in the current season.
    Returns a list of (year, score) tuples sorted ascending by score (best first).
    """
    if current_cum.empty or historical_cdd.empty:
        return []
    n_days = len(current_cum)
    if n_days < 5:
        return []
    curr_y = int(current_cum['date'].dt.year.iloc[0])
    curr_vals = current_cum['cumulative_cdd'].values
    scores = []
    for year in range(HIST_START_YEAR, HIST_END_YEAR + 1):
        if year == curr_y:
            continue
        hist_cum = compute_cumulative(historical_cdd, year)
        if hist_cum.empty or len(hist_cum) < n_days:
            continue
        hist_vals = hist_cum['cumulative_cdd'].iloc[:n_days].values
        rmse = np.sqrt(np.mean((curr_vals - hist_vals) ** 2))
        if n_days > 1:
            rate_rmse = np.sqrt(np.mean((np.diff(curr_vals) - np.diff(hist_vals)) ** 2))
            score = 0.7 * rmse + 0.3 * rate_rmse
        else:
            score = float(rmse)
        scores.append((year, score))
    scores.sort(key=lambda x: x[1])
    return scores[:n_similar]


# ─── Watershed analysis ──────────────────────────────────────────────────────────

# Bounding box derived from three_gorges_catchment.geojson (actual HYDROBASINS catchment)
THREE_GORGES_BOUNDS = {'lat_min': 24.5, 'lat_max': 35.9, 'lon_min': 90.5, 'lon_max': 111.2}
THREE_GORGES_DAM    = (111.0056, 30.8233)   # (lon, lat) — exact dam coordinates


@st.cache_data(ttl=1800, show_spinner=False)
def load_watershed_precip(region_name: str):
    """Return (hist_df, fcst_df, clim_df) area-averaged precipitation for a watershed.

    hist_df : current-year daily area-average precip   (date, precipitation)
    fcst_df : ECMWF-ENS 14-day forecast                (date, precipitation)
    clim_df : ERA5 2000–2024 day-of-year climatology   (doy, mean_precip, std_precip)
    """
    BOUNDS_MAP = {
        'Three Gorges': THREE_GORGES_BOUNDS,
    }
    if region_name not in BOUNDS_MAP:
        empty = pd.DataFrame()
        return empty, empty, empty

    b  = BOUNDS_MAP[region_name]
    cy = datetime.now().year
    box = (f"latitude  BETWEEN {b['lat_min']} AND {b['lat_max']} "
           f"AND longitude BETWEEN {b['lon_min']} AND {b['lon_max']}")

    hist_q = f"""
    SELECT CAST(delivery_start AS DATE) as date, AVG(value) as precipitation
    FROM {TABLE_PRECIP_HIST}
    WHERE model = '{MODEL_HIST}' AND curve_name = '{CURVE_PRECIP_HIST}'
      AND CAST(delivery_start AS DATE) >= '{cy}-01-01'
      AND {box}
    GROUP BY CAST(delivery_start AS DATE) ORDER BY date
    """
    fcst_q = f"""
    SELECT CAST(delivery_start AS DATE) as date, AVG(value) as precipitation
    FROM {TABLE_PRECIP_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_PRECIP_FCST}'
      AND delivery_start >= CURRENT_DATE()
      AND {box}
    GROUP BY CAST(delivery_start AS DATE) ORDER BY date
    """
    clim_q = f"""
    SELECT DAYOFYEAR(delivery_start) as doy,
           AVG(value) as mean_precip, STDDEV(value) as std_precip
    FROM {TABLE_PRECIP_HIST}
    WHERE model = '{MODEL_PRECIP_CLIM}' AND curve_name = '{CURVE_PRECIP_CLIM}'
      AND YEAR(delivery_start) BETWEEN {HIST_START_YEAR} AND {HIST_END_YEAR}
      AND {box}
    GROUP BY DAYOFYEAR(delivery_start) ORDER BY doy
    """

    hist_df = run_query(hist_q)
    fcst_df = run_query(fcst_q)
    clim_df = run_query(clim_q)

    if not hist_df.empty:
        hist_df['date']          = pd.to_datetime(hist_df['date'])
        hist_df['precipitation'] = hist_df['precipitation'].astype(float)
    if not fcst_df.empty:
        fcst_df['date']          = pd.to_datetime(fcst_df['date'])
        fcst_df['precipitation'] = fcst_df['precipitation'].astype(float)
    if not clim_df.empty:
        clim_df['doy']           = clim_df['doy'].astype(int)
        clim_df['mean_precip']   = clim_df['mean_precip'].astype(float)
        clim_df['std_precip']    = clim_df['std_precip'].fillna(0).astype(float)

    _e = lambda cols: pd.DataFrame(columns=cols)
    return (
        hist_df if not hist_df.empty else _e(['date', 'precipitation']),
        fcst_df if not fcst_df.empty else _e(['date', 'precipitation']),
        clim_df if not clim_df.empty else _e(['doy', 'mean_precip', 'std_precip']),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_gatun_lake_levels():
    """Download Gatun Lake observed levels + official ACP forecast.

    Source: Panama Canal Authority — https://evtms-rpts.pancanal.com/eng/h2o/index.html
    Returns (hist_df, proj_df) where both have columns: date (datetime), level_ft (float).
    hist_df contains all historical observations; proj_df contains the official projection.
    """
    import io
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    HIST_URL = ("https://evtms-rpts.pancanal.com/eng/h2o/"
                "Download_Gatun_Lake_Water_Level_History.csv")
    PROJ_URL = ("https://evtms-rpts.pancanal.com/eng/h2o/"
                "Gatun_Water_Level_Projection.csv")
    HEADERS  = {"User-Agent": "Mozilla/5.0 (compatible; CDD-Dashboard/1.0)"}

    def _fetch(url):
        r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        return r.text

    # ── Historical levels ──────────────────────────────────────────────────────
    hist_raw = _fetch(HIST_URL)
    hist_df  = pd.read_csv(io.StringIO(hist_raw))
    hist_df.columns = [c.strip() for c in hist_df.columns]

    date_col  = next((c for c in hist_df.columns if any(k in c.upper() for k in ['DATE', 'FECHA'])),
                     hist_df.columns[0])
    level_col = next((c for c in hist_df.columns if any(k in c.upper() for k in ['LEVEL', 'FEET', 'NIVEL'])),
                     hist_df.columns[1])

    hist_df[date_col]  = pd.to_datetime(hist_df[date_col],  errors='coerce')
    hist_df[level_col] = pd.to_numeric(hist_df[level_col],  errors='coerce')
    hist_df = (hist_df[[date_col, level_col]]
               .dropna()
               .rename(columns={date_col: 'date', level_col: 'level_ft'})
               .query('level_ft > 60')
               .sort_values('date')
               .reset_index(drop=True))

    # ── Official ACP projection ────────────────────────────────────────────────
    proj_raw = _fetch(PROJ_URL)
    proj_df  = None
    for skip in [2, 1, 0]:
        try:
            candidate = pd.read_csv(io.StringIO(proj_raw), skiprows=skip)
            candidate.columns = [c.strip() for c in candidate.columns]
            if len(candidate) >= 3 and len(candidate.columns) >= 2:
                proj_df = candidate
                break
        except Exception:
            continue

    if proj_df is not None:
        date_col  = next((c for c in proj_df.columns if 'date' in c.lower()), proj_df.columns[0])
        level_col = next((c for c in proj_df.columns if any(k in c.lower() for k in ['level', 'water', 'gatun'])),
                         proj_df.columns[1])
        proj_df[date_col]  = pd.to_datetime(proj_df[date_col],  errors='coerce', dayfirst=False)
        proj_df[level_col] = pd.to_numeric(proj_df[level_col],  errors='coerce')
        proj_df = (proj_df[[date_col, level_col]]
                   .dropna()
                   .rename(columns={date_col: 'date', level_col: 'level_ft'})
                   .sort_values('date')
                   .reset_index(drop=True))
    else:
        proj_df = pd.DataFrame(columns=['date', 'level_ft'])

    return hist_df, proj_df


# ─── Hurricane / Tropical Cyclone data ───────────────────────────────────────────

def _kt_to_category(wind_kt: int) -> str:
    """Saffir-Simpson / tropical cyclone category from 1-minute sustained wind (kt)."""
    if wind_kt < 34:    return 'TD'
    elif wind_kt < 64:  return 'TS'
    elif wind_kt < 83:  return 'Cat 1'
    elif wind_kt < 96:  return 'Cat 2'
    elif wind_kt < 113: return 'Cat 3'
    elif wind_kt < 137: return 'Cat 4'
    else:               return 'Cat 5'


def _parse_nhc_forecast_positions(text: str) -> list:
    """Parse 5-day forecast positions from NHC or JTWC forecast advisory text.

    Handles both NHC style:
        INIT  01/2100Z 25.6N  73.5W   115 KT
         12H  02/0900Z 26.9N  75.5W   120 KT
    and JTWC style (space before HR):
         12 HR  23/1800Z   19.4N 123.2E  110 KT
         24 HR  24/0600Z   20.7N 121.0E  110 KT
    """
    import re
    pattern = re.compile(
        r'(INIT|\d+\s*H[RS]?)\s+\d+/\d{4}Z\s+(\d+\.?\d*)\s*([NS])\s+(\d+\.?\d*)\s*([EW])\s+(\d+)\s+KT',
        re.IGNORECASE,
    )
    out = []
    for m in pattern.finditer(text):
        step, lat_v, lat_h, lon_v, lon_h, wind = m.groups()
        step_clean = re.sub(r'[^0-9]', '', step)
        hours = 0 if step.upper().strip() == 'INIT' else (int(step_clean) if step_clean else 0)
        lat = float(lat_v) * (1 if lat_h.upper() == 'N' else -1)
        lon = float(lon_v) * (1 if lon_h.upper() == 'E' else -1)
        out.append({'hours': hours, 'lat': lat, 'lon': lon, 'wind_kt': int(wind)})
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def load_hurricane_data() -> tuple:
    """Fetch active tropical cyclones from NHC + JTWC, with IBTrACS fallback.

    Returns (storms, sources) where:
      storms  — list of storm dicts (id, name, basin, lat, lon, wind_kt,
                pressure_mb, category, classification, last_update, forecast_track)
      sources — dict {'nhc': str, 'jtwc': str, 'ibtracs': str} with 'ok' or error msg
    """
    import re
    import io
    from xml.etree import ElementTree as ET
    import requests as _req

    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CDD-Dashboard/1.0)"}
    storms: list = []
    sources: dict = {'nhc': 'skipped', 'jtwc': 'skipped', 'ibtracs': 'skipped'}

    # ── NHC: Atlantic + Eastern Pacific ────────────────────────────────────────
    try:
        r = _req.get("https://www.nhc.noaa.gov/CurrentStorms.json",
                     headers=HEADERS, timeout=20)
        if r.status_code == 200:
            for s in r.json().get('activeStorms', []):
                sid      = s.get('id', '')
                wind_kt  = int(s.get('intensity', 0))
                storm = {
                    'id':             sid,
                    'name':           s.get('name', 'Unknown'),
                    'basin':          'E.Pacific' if sid.startswith('ep') else 'Atlantic',
                    'lat':            s.get('latitudeNumeric',  0.0),
                    'lon':            s.get('longitudeNumeric', 0.0),
                    'wind_kt':        wind_kt,
                    'pressure_mb':    s.get('pressure', 'N/A'),
                    'category':       _kt_to_category(wind_kt),
                    'classification': s.get('classification', 'TD'),
                    'last_update':    s.get('lastUpdate', ''),
                    'forecast_track': [],
                }
                fcst_url = (s.get('forecastAdvisory') or {}).get('product', '')
                if fcst_url:
                    try:
                        fr = _req.get(fcst_url, headers=HEADERS, timeout=15)
                        if fr.status_code == 200:
                            storm['forecast_track'] = _parse_nhc_forecast_positions(fr.text)
                    except Exception:
                        pass
                storms.append(storm)
            sources['nhc'] = 'ok'
        else:
            sources['nhc'] = f'HTTP {r.status_code}'
    except Exception as e:
        sources['nhc'] = str(e)

    # ── JTWC / RSMC products via NOAA tgftp mirror (accessible .gov domain) ─────
    # NOAA mirrors JTWC warning products at tgftp.nws.noaa.gov.
    # We fetch the /wt/ directory listing, filter for warning files (3x series),
    # then parse position + forecast track from each product text.
    TGFTP_WMO = "https://tgftp.nws.noaa.gov/data/raw/wt/"
    # Basin codes in WMO product IDs → human-readable name
    JTWC_BASIN = {'pn': 'W.Pacific', 'xs': 'S.Pacific', 'xo': 'S.Pacific',
                  'io': 'Indian Ocean', 'in': 'Indian Ocean'}
    # Regex for Apache directory listing links — filenames end with ..txt (two dots)
    # e.g. wtpn31.pgtw..txt, wtxs31.pgtw..txt, wtin31.dems..txt
    FILE_RE = re.compile(
        r'href="(wt[a-z]{2}\d{2,3}\.[a-z]{4}\.\.txt)"',
        re.IGNORECASE,
    )

    jtwc_storms_added = 0
    seen_storm_ids: set = set()
    try:
        dr = _req.get(TGFTP_WMO, headers=HEADERS, timeout=20)
        if dr.status_code == 200:
            filenames = FILE_RE.findall(dr.text)
            # Include warning series (30-39) AND formation/special products (20-29);
            # any file without a T000 position line is silently skipped below.
            warning_files = []
            for fn in filenames:
                num_m = re.search(r'wt[a-z]{2}(\d{2,3})\.', fn)
                if num_m and 20 <= int(num_m.group(1)) <= 59:
                    warning_files.append(fn)

            for filename in warning_files:
                basin_code = re.match(r'wt([a-z]{2})', filename)
                basin = JTWC_BASIN.get(basin_code.group(1) if basin_code else '', '')
                if not basin:
                    continue

                try:
                    tr = _req.get(TGFTP_WMO + filename, headers=HEADERS, timeout=20)
                    if tr.status_code != 200:
                        continue
                    pt = tr.text
                except Exception:
                    continue

                # JTWC GENTEXT advisory format (current as of 2025+):
                #   WMO heading:     "WTPN31 PGTW 250900"  (DDHHMM, no Z)
                #   Current pos:     "250600Z --- NEAR 24.8N 126.0E"
                #                    under the "WARNING POSITION:" label
                #   Forecast blocks: "12 HRS, VALID AT:\n251800Z --- 26.7N 127.2E"
                #
                # Stale files (past storms) stay on tgftp indefinitely.
                # Two guards keep only fresh, active bulletins:
                #   1. Skip FINAL advisories (storm has dissipated).
                #   2. Skip bulletins whose WMO heading day ≠ today or yesterday UTC.

                if re.search(r'FINAL\s+(?:WARNING|ADVISORY)', pt, re.IGNORECASE):
                    continue

                hdr_m = re.search(r'^WT[A-Z]{2}\d+\s+[A-Z]+\s+(\d{2})\d{4}', pt, re.MULTILINE)
                if not hdr_m:
                    continue  # can't verify bulletin age — skip conservatively
                bulletin_day = int(hdr_m.group(1))
                now_utc = datetime.utcnow()
                if bulletin_day not in {now_utc.day, (now_utc - timedelta(days=1)).day}:
                    continue

                # Parse current position and its DTG:
                # "250600Z --- NEAR 24.8N 126.0E"
                # Capture DTG groups (day, hhmm) so we can age-check the advisory.
                wp_m = re.search(
                    r'WARNING\s+POSITION:\s*\n\s*(\d{2})(\d{4})Z\s+---\s+(?:NEAR\s+)?(\d+\.?\d*)\s*([NS])\s+(\d+\.?\d*)\s*([EW])',
                    pt,
                )
                if not wp_m:
                    continue

                # Age check: JTWC updates every 6 h; an active storm has a fix within
                # the last 6 h. If the WARNING POSITION is > 18 h old the storm has
                # missed at least 2 advisory cycles and is no longer being tracked.
                pos_day_s, pos_hhmm, lat_s, lat_h, lon_s, lon_h = wp_m.groups()
                try:
                    pos_dt = now_utc.replace(day=int(pos_day_s),
                                             hour=int(pos_hhmm[:2]),
                                             minute=int(pos_hhmm[2:]),
                                             second=0, microsecond=0)
                    if pos_dt > now_utc + timedelta(hours=1):
                        # day belongs to previous month
                        m = now_utc.month - 1 or 12
                        y = now_utc.year if now_utc.month > 1 else now_utc.year - 1
                        pos_dt = pos_dt.replace(year=y, month=m)
                    if (now_utc - pos_dt).total_seconds() > 18 * 3600:
                        continue  # last fix too old — storm no longer active
                except (ValueError, OverflowError):
                    pass  # unparseable DTG; proceed
                lat = float(lat_s) * (1 if lat_h == 'N' else -1)
                lon = float(lon_s) * (1 if lon_h == 'E' else -1)

                # Wind: "MAX SUSTAINED WINDS - 050 KT"
                wm = re.search(r'MAX\s+SUSTAINED\s+WINDS\s*-\s*(\d+)\s+KT', pt, re.IGNORECASE)
                wind_kt = int(wm.group(1)) if wm else 0

                # Pressure: "MINIMUM CENTRAL PRESSURE AT 230600Z IS 986 MB"
                pm = re.search(
                    r'CENTRAL\s+PRESSURE\s+AT\s+\d+Z\s+IS\s+(\d{3,4})\s+MB', pt, re.IGNORECASE,
                )
                pressure = pm.group(1) if pm else 'N/A'

                # Storm number "07W" / "27P" — dedup key
                id_m = re.search(
                    r'(?:TYPHOON|TROPICAL\s+STORM|CYCLONE|DEPRESSION)\s+(\d+[A-Z])',
                    pt, re.IGNORECASE,
                )
                storm_num = id_m.group(1).upper() if id_m else filename[:10]
                if storm_num in seen_storm_ids:
                    continue
                seen_storm_ids.add(storm_num)

                # Storm name: "TROPICAL STORM 07W (MEKKHALA)"
                nm = re.search(
                    r'(?:TYPHOON|TROPICAL\s+STORM|CYCLONE|DEPRESSION)\s+\d+[A-Z]\s+\(([A-Z]{2,})\)',
                    pt, re.IGNORECASE,
                )
                name = nm.group(1).capitalize() if nm else storm_num

                # Forecast track from "X HRS, VALID AT:" blocks
                FCST_RE = re.compile(
                    r'(\d+)\s+HRS?,\s+VALID\s+AT:[^\n]*\n\s*\d{6}Z\s+---\s+(?:NEAR\s+)?(\d+\.?\d*)\s*([NS])\s+(\d+\.?\d*)\s*([EW])',
                    re.MULTILINE,
                )
                forecast_track = [
                    {
                        'hours':   int(h),
                        'lat':     float(la) * (1 if lah == 'N' else -1),
                        'lon':     float(lo) * (1 if loh == 'E' else -1),
                        'wind_kt': wind_kt,
                    }
                    for h, la, lah, lo, loh in FCST_RE.findall(pt)
                ]

                classification = 'TY' if wind_kt >= 64 else ('TS' if wind_kt >= 34 else 'TD')
                storms.append({
                    'id':             f'jtwc-{storm_num.lower()}',
                    'name':           name,
                    'basin':          basin,
                    'lat':            lat,
                    'lon':            lon,
                    'wind_kt':        wind_kt,
                    'pressure_mb':    pressure,
                    'category':       _kt_to_category(wind_kt),
                    'classification': classification,
                    'last_update':    '',
                    'forecast_track': forecast_track,
                })
                jtwc_storms_added += 1

            sources['jtwc'] = 'ok'
        else:
            sources['jtwc'] = f'HTTP {dr.status_code}'
    except Exception as e:
        sources['jtwc'] = str(e)

    # ── IBTrACS ACTIVE: fills gaps that tgftp parsing missed ────────────────────
    # Uses position-proximity dedup (2°) to avoid duplicating JTWC storms.
    # The ACTIVE file is only updated during active storm seasons; skip if
    # Last-Modified shows it hasn't changed in > 24 h (quiet season).
    if True:
        _ibtracs_ok = True
        try:
            import pandas as _pd
            import email.utils as _eu
            IBTRACS_URL = (
                "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
                "stewardship-ibtracs/v04r00/access/csv/ibtracs.ACTIVE.list.v04r00.csv"
            )
            # HEAD first to check freshness without downloading the full file
            hd = _req.head(IBTRACS_URL, headers=HEADERS, timeout=10)
            lm = hd.headers.get('Last-Modified', '')
            if lm:
                lm_dt = _eu.parsedate_to_datetime(lm)
                import datetime as _dt2
                lm_utc = lm_dt.astimezone(_dt2.timezone.utc).replace(tzinfo=None)
                age_h = (datetime.utcnow() - lm_utc).total_seconds() / 3600
                if age_h > 24:
                    sources['ibtracs'] = f'skipped ({int(age_h)}h stale)'
                    _ibtracs_ok = False
        except Exception:
            _ibtracs_ok = True  # if HEAD fails, try the GET anyway
        try:
            if _ibtracs_ok:
                ir = _req.get(IBTRACS_URL, headers=HEADERS, timeout=30)
            if _ibtracs_ok and ir.status_code == 200:
                df = _pd.read_csv(io.StringIO(ir.text), skiprows=[1], low_memory=False)
                df['ISO_TIME'] = _pd.to_datetime(df['ISO_TIME'], errors='coerce')
                df = df.dropna(subset=['ISO_TIME', 'LAT', 'LON'])

                # TRACK_TYPE is the definitive active-storm indicator in IBTrACS:
                #   'provisional' = currently tracked, real-time unfinalized data
                #   'main'        = finalized historical record
                # The ACTIVE file can briefly retain recently-dissipated 'main' tracks,
                # which is why date comparisons alone keep returning historical storms.
                if 'TRACK_TYPE' in df.columns:
                    prov = df[df['TRACK_TYPE'].str.strip().str.lower() == 'provisional']
                    if not prov.empty:
                        df = prov

                # Most recent record per storm (after TRACK_TYPE filter)
                latest = df.sort_values('ISO_TIME').groupby('SID').last().reset_index()

                # Secondary guard: drop anything last seen more than 48 h ago.
                # ISO_TIME is tz-naive; cutoff_ts built from naive utcnow() matches it.
                import datetime as _dt_m
                cutoff_ts = _pd.Timestamp(_dt_m.datetime.utcnow() - _dt_m.timedelta(hours=48))
                latest = latest[latest['ISO_TIME'] >= cutoff_ts]

                if latest.empty:
                    sources['ibtracs'] = 'ok (no current storms)'

                BASIN_MAP = {
                    'WP': 'W.Pacific', 'EP': 'E.Pacific', 'NA': 'Atlantic',
                    'NI': 'Indian Ocean', 'SI': 'Indian Ocean',
                    'SP': 'S.Pacific', 'SA': 'S.Atlantic',
                }
                NHC_BASINS = {'Atlantic', 'E.Pacific'}

                for _, row in latest.iterrows():
                    basin = BASIN_MAP.get(str(row.get('BASIN', '')).strip(), '')
                    if not basin or basin in NHC_BASINS:
                        continue

                    name = str(row.get('NAME', '')).strip()
                    if not name or name.upper() in ('UNNAMED', 'NOT_NAMED', 'NAN'):
                        name = f"TC-{str(row.get('BASIN', ''))}"

                    try:
                        lat = float(row['LAT'])
                        lon = float(row['LON'])
                        # WMO_WIND is blank for many basins; fall back to USA_WIND
                        for wcol in ('WMO_WIND', 'USA_WIND'):
                            raw = str(row.get(wcol, '')).strip()
                            if raw and raw.lower() not in ('', ' ', 'nan'):
                                wind_kt = int(float(raw))
                                break
                        else:
                            wind_kt = 0
                        for pcol in ('WMO_PRES', 'USA_PRES'):
                            raw = str(row.get(pcol, '')).strip()
                            if raw and raw.lower() not in ('', ' ', 'nan'):
                                pressure_ibt = raw
                                break
                        else:
                            pressure_ibt = 'N/A'
                    except (ValueError, TypeError):
                        continue

                    if abs(lat) > 90 or abs(lon) > 180:
                        continue

                    # Skip if JTWC already provided this storm (position within 2°)
                    already_have = any(
                        abs(s['lat'] - lat) < 2.0 and abs(s['lon'] - lon) < 2.0
                        for s in storms
                    )
                    if already_have:
                        continue

                    storms.append({
                        'id':             str(row.get('SID', f'ibt-{name}')),
                        'name':           name.capitalize(),
                        'basin':          basin,
                        'lat':            lat,
                        'lon':            lon,
                        'wind_kt':        wind_kt,
                        'pressure_mb':    pressure_ibt,
                        'category':       _kt_to_category(wind_kt),
                        'classification': 'TY' if wind_kt >= 64 else ('TS' if wind_kt >= 34 else 'TD'),
                        'last_update':    str(row['ISO_TIME'])[:16],
                        'forecast_track': [],
                    })
                sources['ibtracs'] = 'ok'
            elif _ibtracs_ok:
                sources['ibtracs'] = f'HTTP {ir.status_code}'
        except Exception as e:
            sources['ibtracs'] = str(e)

    # ── ECMWF AIFS TC tracks (requires: pip install eccodes) ─────────────────
    # data.ecmwf.int publishes AIFS (AI model) tropical forecast BUFR files
    # with no authentication. Each BUFR message = one TC; arrays inside give
    # analysis + 360h forecast positions.
    try:
        import eccodes as _ec  # type: ignore
        import datetime as _dt
        import tempfile as _tmp
        import os as _os

        # BUFR missing-value sentinels used by eccodes
        _DBL_MISS = _ec.CODES_MISSING_DOUBLE
        _INT_MISS = _ec.CODES_MISSING_LONG

        def _ec_get(mid, key, default=None):
            try:
                v = _ec.codes_get(mid, key)
                return default if (v == _DBL_MISS or v == _INT_MISS) else v
            except Exception:
                return default

        def _ec_arr(mid, key):
            try:
                return [v for v in _ec.codes_get_array(mid, key)
                        if v != _DBL_MISS and v != _INT_MISS and abs(v) < 1e30]
            except Exception:
                return []

        today_str = _dt.date.today().strftime('%Y%m%d')
        # Try 00z first, fall back to 12z (previous run)
        bufr_url = None
        for run_h, run_sfx in (('00z', '00'), ('12z', '12')):
            _u = (f"https://data.ecmwf.int/forecasts/{today_str}/{run_h}/"
                  f"aifs-single/0p25/oper/{today_str}{run_sfx}0000-360h-oper-tf.bufr")
            try:
                _r = _req.get(_u, headers=HEADERS, timeout=5)
                if _r.status_code == 200:
                    bufr_url = _u
                    bufr_content = _r.content
                    break
            except Exception:
                pass

        if bufr_url is None:
            sources['ecmwf_aifs'] = 'BUFR file not found (try again after 06:00 UTC)'
        else:
            tmp_fd, tmp_path = _tmp.mkstemp(suffix='.bufr')
            aifs_added = 0
            try:
                _os.write(tmp_fd, bufr_content)
                _os.close(tmp_fd)

                with open(tmp_path, 'rb') as _f:
                    while True:
                        msg = _ec.codes_bufr_new_from_file(_f)
                        if msg is None:
                            break
                        try:
                            _ec.codes_set(msg, 'unpack', 1)

                            # Storm name — try several key variants
                            name_raw = (
                                _ec_get(msg, 'stormName')
                                or _ec_get(msg, 'nameOfTheCyclonicSystem')
                                or _ec_get(msg, 'stormIdentifier')
                                or 'UNKNOWN'
                            )
                            name_str = str(name_raw).strip().capitalize()

                            # Lat/lon arrays (analysis + forecast positions)
                            lats = _ec_arr(msg, 'latitude')
                            lons = _ec_arr(msg, 'longitude')
                            if not lats or not lons or len(lats) < 1:
                                continue

                            # Wind speed arrays — BUFR may store in m/s
                            winds_raw = (
                                _ec_arr(msg, 'maximumWindSpeed')
                                or _ec_arr(msg, 'windSpeed')
                                or _ec_arr(msg, 'maximumWindGustSpeed')
                            )
                            # Convert m/s → kt if values look like m/s (< 150)
                            if winds_raw and max(winds_raw) < 150:
                                winds_kt = [w / 0.514444 for w in winds_raw]
                            else:
                                winds_kt = list(winds_raw) if winds_raw else []

                            # Pressure arrays (Pa → hPa)
                            pres_raw = (
                                _ec_arr(msg, 'minimumPressureAtMeanSeaLevel')
                                or _ec_arr(msg, 'pressure')
                                or _ec_arr(msg, 'pressureReducedToMeanSeaLevel')
                            )
                            pres_hpa = [p / 100 if p > 10000 else p for p in pres_raw]

                            # Time offsets in hours (index 0 = analysis = 0h)
                            time_offsets = _ec_arr(msg, 'timePeriod') or list(range(0, len(lats) * 12, 12))

                            lat0 = float(lats[0])
                            lon0 = float(lons[0])
                            wkt0 = int(winds_kt[0]) if winds_kt else 0
                            pr0 = f"{int(pres_hpa[0])}" if pres_hpa else 'N/A'

                            track = []
                            for i in range(1, len(lats)):
                                if abs(lats[i]) > 90 or abs(lons[i]) > 180:
                                    continue
                                track.append({
                                    'hours':   int(time_offsets[i]) if i < len(time_offsets) else i * 12,
                                    'lat':     float(lats[i]),
                                    'lon':     float(lons[i]),
                                    'wind_kt': int(winds_kt[i]) if i < len(winds_kt) else 0,
                                })

                            storms.append({
                                'id':             f'aifs-{name_str.lower()[:8]}',
                                'name':           f'{name_str} ★AIFS',
                                'basin':          'AI (ECMWF)',
                                'lat':            lat0,
                                'lon':            lon0,
                                'wind_kt':        wkt0,
                                'pressure_mb':    pr0,
                                'category':       _kt_to_category(wkt0),
                                'classification': 'AI',
                                'last_update':    'ECMWF AIFS',
                                'forecast_track': track,
                            })
                            aifs_added += 1

                        finally:
                            _ec.codes_release(msg)

            finally:
                _os.unlink(tmp_path)

            sources['ecmwf_aifs'] = f'ok ({aifs_added} TC tracks from AIFS AI model)'

    except ImportError:
        sources['ecmwf_aifs'] = 'eccodes not installed — add: pip install eccodes'
    except Exception as e:
        sources['ecmwf_aifs'] = str(e)

    return storms, sources
