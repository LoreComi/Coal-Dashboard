"""Data access layer — Coal Desk CDD Dashboard.

Uses the Databricks SQL Statement Execution REST API with the user's
forwarded access token (same pattern as the Morning Met dashboard).
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
)

# ─── Warehouse config ───────────────────────────────────────────────────────────
_raw_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_HOST = _raw_host if _raw_host.startswith("https://") else f"https://{_raw_host}"
WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "").split("/")[-1]


# ─── Low-level query (databricks-sdk + user token fallback) ──────────────────────

def _get_token() -> str:
    """Get the best available token: user forwarded token or SDK default."""
    # Try user-forwarded token first (has user's permissions)
    try:
        user_token = st.context.headers.get("x-forwarded-access-token")
        if user_token:
            return user_token
    except Exception:
        pass
    # Fallback: use the app service principal's token via SDK
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    return w.config.authenticate()


def run_query(query: str) -> pd.DataFrame:
    """Execute SQL via the Databricks Statement Execution API."""
    # Try user token first, then fall back to SP token
    user_token = None
    try:
        user_token = st.context.headers.get("x-forwarded-access-token")
    except Exception:
        pass

    if user_token:
        headers = {"Authorization": f"Bearer {user_token}", "Content-Type": "application/json"}
    else:
        # Use databricks-sdk default auth (app service principal)
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        headers = {"Content-Type": "application/json"}
        headers.update(w.api_client.default_headers)

    resp = requests.post(
        f"{DATABRICKS_HOST}/api/2.0/sql/statements/",
        headers=headers,
        json={"warehouse_id": WAREHOUSE_ID, "statement": query, "wait_timeout": "50s"},
        timeout=60,
    )

    # If user token fails with 403, retry with SP token
    if resp.status_code == 403 and user_token:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        sp_headers = {"Content-Type": "application/json"}
        sp_headers.update(w.api_client.default_headers)
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
        msg = status.get("error", {}).get("message", "Unknown error")
        raise RuntimeError(f"Query failed: {msg}")

    columns = [col["name"] for col in data.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = data.get("result", {}).get("data_array", [])
    return pd.DataFrame(rows, columns=columns)


COAL_DESK_SCHEMA = "dna_snbx_weather.coal_desk"


# ─── Coordinate helpers ──────────────────────────────────────────────────────────

def _coord_filter(cities: list[str]) -> str:
    """Build SQL WHERE clause for lat/lon pairs."""
    parts = []
    for city in cities:
        loc = CITY_LOCATIONS[city]
        parts.append(f"(latitude = {loc['latitude']} AND longitude = {loc['longitude']})")
    return " OR ".join(parts)


def _coord_to_city_map(cities: list[str]) -> dict[tuple, str]:
    return {
        (CITY_LOCATIONS[c]['latitude'], CITY_LOCATIONS[c]['longitude']): c
        for c in cities
    }


# ─── Data loaders ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_historical(region: str) -> pd.DataFrame:
    """Load historical ERA5 t_mean for all cities in a region (from 2000)."""
    cities = REGION_MAP[region]
    cf = _coord_filter(cities)
    query = f"""
    SELECT CAST(delivery_start AS DATE) as date, value as temperature,
           latitude, longitude
    FROM {TABLE_HIST}
    WHERE model = '{MODEL_HIST}'
      AND curve_name = '{CURVE_HIST}'
      AND delivery_start >= '{HIST_START_YEAR}-01-01'
      AND ({cf})
    ORDER BY delivery_start
    """
    df = run_query(query)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df['temperature'] = df['temperature'].astype(float)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    lookup = _coord_to_city_map(cities)
    df['city'] = df.apply(lambda r: lookup.get((r['latitude'], r['longitude']), '?'), axis=1)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(region: str) -> pd.DataFrame:
    """Load latest ECMWF-ENS t_mean forecast for a region."""
    cities = REGION_MAP[region]
    cf = _coord_filter(cities)
    query = f"""
    SELECT CAST(delivery_start AS DATE) as date, value as temperature,
           latitude, longitude
    FROM {TABLE_FCST}
    WHERE model = '{MODEL_FCST}'
      AND curve_name = '{CURVE_FCST}'
      AND delivery_start >= CURRENT_DATE()
      AND ({cf})
    ORDER BY delivery_start
    """
    df = run_query(query)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df['temperature'] = df['temperature'].astype(float)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    lookup = _coord_to_city_map(cities)
    df['city'] = df.apply(lambda r: lookup.get((r['latitude'], r['longitude']), '?'), axis=1)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_city_timeseries(city: str, lookback_days: int = 90) -> pd.DataFrame:
    """Load historical + forecast for a single city."""
    loc = CITY_LOCATIONS[city]
    lat, lon = loc['latitude'], loc['longitude']
    since = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    hist_q = f"""
    SELECT CAST(delivery_start AS DATE) as date, value as temperature
    FROM {TABLE_HIST}
    WHERE model = '{MODEL_HIST}' AND curve_name = '{CURVE_HIST}'
      AND latitude = {lat} AND longitude = {lon}
      AND delivery_start >= '{since}'
    ORDER BY delivery_start
    """
    fcst_q = f"""
    SELECT CAST(delivery_start AS DATE) as date, AVG(value) as temperature
    FROM {TABLE_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_FCST}'
      AND latitude = {lat} AND longitude = {lon}
      AND delivery_start >= CURRENT_DATE()
    GROUP BY CAST(delivery_start AS DATE)
    ORDER BY date
    """
    h = run_query(hist_q)
    f = run_query(fcst_q)
    frames = []
    for df in (h, f):
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['temperature'] = df['temperature'].astype(float)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['date', 'temperature'])
    out = pd.concat(frames, ignore_index=True).drop_duplicates('date', keep='first').sort_values('date')
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_anomalies(target_date) -> pd.DataFrame:
    """Compute forecast anomaly vs climatology for all cities on a given date."""
    all_cities = list(CITY_LOCATIONS.keys())
    cf = _coord_filter(all_cities)
    target_str = str(target_date)
    doy = pd.Timestamp(target_date).day_of_year

    fcst_q = f"""
    SELECT CAST(delivery_start AS DATE) as date, AVG(value) as temperature,
           latitude, longitude
    FROM {TABLE_FCST}
    WHERE model = '{MODEL_FCST}' AND curve_name = '{CURVE_FCST}'
      AND CAST(delivery_start AS DATE) = '{target_str}'
      AND ({cf})
    GROUP BY CAST(delivery_start AS DATE), latitude, longitude
    """
    clim_q = f"""
    SELECT AVG(value) as climatology, latitude, longitude
    FROM {TABLE_HIST}
    WHERE model = '{MODEL_HIST}' AND curve_name = '{CURVE_HIST}'
      AND DAYOFYEAR(delivery_start) = {doy}
      AND YEAR(delivery_start) BETWEEN {HIST_START_YEAR} AND {HIST_END_YEAR}
      AND ({cf})
    GROUP BY latitude, longitude
    """
    fcst_df = run_query(fcst_q)
    clim_df = run_query(clim_q)
    if fcst_df.empty or clim_df.empty:
        return pd.DataFrame()

    for df in (fcst_df, clim_df):
        for col in ('latitude', 'longitude'):
            df[col] = df[col].astype(float)
    fcst_df['temperature'] = fcst_df['temperature'].astype(float)
    clim_df['climatology'] = clim_df['climatology'].astype(float)

    merged = fcst_df.merge(clim_df, on=['latitude', 'longitude'], how='inner')
    merged['anomaly'] = merged['temperature'] - merged['climatology']

    lookup = _coord_to_city_map(all_cities)
    merged['city'] = merged.apply(lambda r: lookup.get((r['latitude'], r['longitude']), '?'), axis=1)
    merged['region'] = merged['city'].map(CITY_TO_REGION)
    return merged



# ─── Pre-computed table loaders (from ingestion pipeline) ──────────────────────

COAL_DESK_SCHEMA = "dna_snbx_weather.coal_desk"


@st.cache_data(ttl=1800, show_spinner=False)
def load_precomputed_cdd() -> pd.DataFrame:
    """Load region-level CDD from ingestion pipeline (ecmwf-ens + vareps)."""
    query = f"SELECT date, region, model, cdd, temperature, n_cities FROM {COAL_DESK_SCHEMA}.coal_desk_cdd ORDER BY date"
    df = run_query(query)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['cdd'] = df['cdd'].astype(float)
        df['temperature'] = df['temperature'].astype(float)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_precomputed_historical() -> pd.DataFrame:
    """Load historical region-level CDD (2000-2024)."""
    query = f"SELECT date, region, cdd, temperature FROM {COAL_DESK_SCHEMA}.coal_desk_cdd_historical ORDER BY date"
    df = run_query(query)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['cdd'] = df['cdd'].astype(float)
        df['temperature'] = df['temperature'].astype(float)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_precomputed_forecasts(parameter: str = None, model: str = None) -> pd.DataFrame:
    """Load city-level forecasts (t_min, t_max, t_mean) from ingestion."""
    conditions = []
    if parameter:
        conditions.append(f"parameter = '{parameter}'")
    if model:
        conditions.append(f"model = '{model}'")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"SELECT date, city, value, parameter, model, label FROM {COAL_DESK_SCHEMA}.coal_desk_forecasts {where} ORDER BY date"
    df = run_query(query)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = df['value'].astype(float)
    return df


# ─── CDD computation ────────────────────────────────────────────────────────────

def compute_region_cdd(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Population-weighted daily CDD for a region."""
    if df.empty:
        return pd.DataFrame(columns=['date', 'cdd'])
    cities = REGION_MAP[region]
    df = df.copy()
    df['cdd'] = (df['temperature'] - BASE_TEMP).clip(lower=0)
    pivot = df.pivot_table(index='date', columns='city', values='cdd', aggfunc='mean')
    available = [c for c in cities if c in pivot.columns]
    if not available:
        return pd.DataFrame(columns=['date', 'cdd'])
    pops = np.array([POPULATION[c] for c in available], dtype=float)
    weights = pops / pops.sum()
    weighted = (pivot[available].values * weights[None, :]).sum(axis=1)
    return pd.DataFrame({'date': pivot.index, 'cdd': weighted}).sort_values('date').reset_index(drop=True)


def compute_cumulative(cdd_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Cumulative CDD from season start for a given year."""
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
    """Historical mean +/- 1 std of cumulative CDD (2000–2024)."""
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


def compute_region_temperature(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Population-weighted daily mean temperature for a region."""
    if df.empty:
        return pd.DataFrame(columns=['date', 'temperature'])
    cities = REGION_MAP[region]
    df = df.copy()
    pivot = df.pivot_table(index='date', columns='city', values='temperature', aggfunc='mean')
    available = [c for c in cities if c in pivot.columns]
    if not available:
        return pd.DataFrame(columns=['date', 'temperature'])
    pops = np.array([POPULATION[c] for c in available], dtype=float)
    weights = pops / pops.sum()
    weighted = (pivot[available].values * weights[None, :]).sum(axis=1)
    return pd.DataFrame({'date': pivot.index, 'temperature': weighted}).sort_values('date').reset_index(drop=True)


def compute_daily_cdd_climatology(region_cdd_df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-year mean and std of daily CDD from historical data (2000–2024)."""
    if region_cdd_df.empty:
        return pd.DataFrame(columns=['day_of_year', 'mean_cdd', 'std_cdd'])
    df = region_cdd_df.copy()
    df['day_of_year'] = df['date'].dt.day_of_year
    df['year'] = df['date'].dt.year
    df = df[df['year'].between(HIST_START_YEAR, HIST_END_YEAR)]
    stats = df.groupby('day_of_year')['cdd'].agg(['mean', 'std']).reset_index()
    stats.columns = ['day_of_year', 'mean_cdd', 'std_cdd']
    stats['std_cdd'] = stats['std_cdd'].fillna(0)
    return stats


def compute_temperature_climatology(region_hist_df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Day-of-year mean and std of region temperature from historical data (2000–2024)."""
    if region_hist_df.empty:
        return pd.DataFrame(columns=['day_of_year', 'mean_temp', 'std_temp'])
    temp_df = compute_region_temperature(region_hist_df, region)
    if temp_df.empty:
        return pd.DataFrame(columns=['day_of_year', 'mean_temp', 'std_temp'])
    temp_df['day_of_year'] = temp_df['date'].dt.day_of_year
    temp_df['year'] = temp_df['date'].dt.year
    temp_df = temp_df[temp_df['year'].between(HIST_START_YEAR, HIST_END_YEAR)]
    stats = temp_df.groupby('day_of_year')['temperature'].agg(['mean', 'std']).reset_index()
    stats.columns = ['day_of_year', 'mean_temp', 'std_temp']
    stats['std_temp'] = stats['std_temp'].fillna(0)
    return stats
