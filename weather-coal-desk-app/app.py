"""Coal Desk Weather / CDD Report — Trader Edition
========================================================
Built for LPG & coal traders. Population-weighted CDD across
57 cities, 16 regions. Data from Databricks Unity Catalog
(dna_prod_silver.meteomatics).

Module layout
-------------
    app.py         ← this file (UI / orchestration)
    _config.py     ← cities, regions, populations, constants
    _data.py       ← SQL & caching layer (REST API)
    _charts.py     ← Plotly figure builders
    _style.py      ← CSS + Plotly dark-navy theme
"""
PLACEHOLDER_SPLIT

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Coal Desk Weather / CDD Report",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# City, Region, and Population Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITY_LOCATIONS = {
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

POPULATION = {
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

REGION_MAP = {
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

BASE_TEMP = 18.0  # CDD base temperature in Celsius
SEASON_START_MONTH = 4
SEASON_START_DAY = 15

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Database Connection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def get_workspace_client():
    """Initialize the Databricks Workspace Client."""
    return WorkspaceClient()


def run_sql_query(query: str, warehouse_id: str = None) -> pd.DataFrame:
    """Execute a SQL query against a Databricks SQL warehouse and return a DataFrame."""
    w = get_workspace_client()
    if warehouse_id is None:
        warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")

    if not warehouse_id:
        # Try to find an available warehouse
        warehouses = list(w.warehouses.list())
        if warehouses:
            warehouse_id = warehouses[0].id
        else:
            st.error("No SQL warehouse available. Set DATABRICKS_WAREHOUSE_ID env var.")
            return pd.DataFrame()

    response = w.statement_execution.execute_statement(
        statement=query,
        warehouse_id=warehouse_id,
        wait_timeout="120s",
    )

    if response.status.state != StatementState.SUCCEEDED:
        st.error(f"Query failed: {response.status.error}")
        return pd.DataFrame()

    # Parse results into DataFrame
    columns = [col.name for col in response.manifest.schema.columns]
    rows = []
    if response.result and response.result.data_array:
        rows = response.result.data_array

    df = pd.DataFrame(rows, columns=columns)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loading Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_coord_filter(cities: list) -> str:
    """Build a SQL WHERE clause to filter by lat/lon pairs for given cities."""
    conditions = []
    for city in cities:
        loc = CITY_LOCATIONS[city]
        conditions.append(
            f"(latitude = {loc['latitude']} AND longitude = {loc['longitude']})"
        )
    return " OR ".join(conditions)


@st.cache_data(ttl=3600)
def load_historical_temperature(region: str, start_year: int = 2000) -> pd.DataFrame:
    """Load historical ERA5 temperature data for all cities in a region."""
    cities = REGION_MAP[region]
    coord_filter = build_coord_filter(cities)

    query = f"""
    SELECT
        CAST(delivery_start AS DATE) as date,
        value as temperature,
        latitude,
        longitude
    FROM dna_prod_silver.meteomatics.temperature
    WHERE model = 'ecmwf-era5'
      AND curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d'
      AND delivery_start >= '{start_year}-01-01'
      AND ({coord_filter})
    ORDER BY delivery_start
    """

    df = run_sql_query(query)
    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df['temperature'] = df['temperature'].astype(float)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)

    # Map lat/lon back to city names
    coord_to_city = {}
    for city in cities:
        loc = CITY_LOCATIONS[city]
        coord_to_city[(loc['latitude'], loc['longitude'])] = city

    df['city'] = df.apply(
        lambda r: coord_to_city.get((r['latitude'], r['longitude']), 'Unknown'), axis=1
    )
    return df


@st.cache_data(ttl=1800)
def load_forecast_temperature(region: str) -> pd.DataFrame:
    """Load the latest ECMWF-ENS forecast for all cities in a region."""
    cities = REGION_MAP[region]
    coord_filter = build_coord_filter(cities)

    query = f"""
    SELECT
        CAST(delivery_start AS DATE) as date,
        value as temperature,
        latitude,
        longitude
    FROM dna_prod_silver.meteomatics.temperature_forecast
    WHERE model = 'ecmwf-ens'
      AND curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d'
      AND ({coord_filter})
      AND delivery_start >= CURRENT_DATE()
    ORDER BY delivery_start
    """

    df = run_sql_query(query)
    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df['temperature'] = df['temperature'].astype(float)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)

    coord_to_city = {}
    for city in cities:
        loc = CITY_LOCATIONS[city]
        coord_to_city[(loc['latitude'], loc['longitude'])] = city

    df['city'] = df.apply(
        lambda r: coord_to_city.get((r['latitude'], r['longitude']), 'Unknown'), axis=1
    )
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CDD Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_region_cdd(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Compute population-weighted CDD for a region from city-level temperature data."""
    if df.empty:
        return pd.DataFrame(columns=['date', 'cdd'])

    cities = REGION_MAP[region]

    # Compute CDD per city
    df = df.copy()
    df['cdd'] = (df['temperature'] - BASE_TEMP).clip(lower=0)

    # Pivot to get one column per city
    pivot = df.pivot_table(index='date', columns='city', values='cdd', aggfunc='mean')

    # Population-weighted average
    available_cities = [c for c in cities if c in pivot.columns]
    if not available_cities:
        return pd.DataFrame(columns=['date', 'cdd'])

    pops = np.array([POPULATION[c] for c in available_cities], dtype=float)
    weights = pops / pops.sum()

    weighted_cdd = (pivot[available_cities].values * weights[None, :]).sum(axis=1)
    result = pd.DataFrame({'date': pivot.index, 'cdd': weighted_cdd})
    result = result.sort_values('date').reset_index(drop=True)
    return result


def compute_cumulative_cdd(cdd_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute cumulative CDD from season start (April 15) for a given year."""
    season_start = pd.Timestamp(year=year, month=SEASON_START_MONTH, day=SEASON_START_DAY)
    season_end = pd.Timestamp(year=year + 1, month=SEASON_START_MONTH, day=SEASON_START_DAY - 1)

    mask = (cdd_df['date'] >= season_start) & (cdd_df['date'] <= season_end)
    season = cdd_df[mask].copy().sort_values('date')

    if season.empty:
        return pd.DataFrame(columns=['date', 'day_of_season', 'cumulative_cdd'])

    season['cumulative_cdd'] = season['cdd'].cumsum()
    season['day_of_season'] = range(1, len(season) + 1)
    return season[['date', 'day_of_season', 'cumulative_cdd']]


def compute_historical_normal(cdd_df: pd.DataFrame, start_year: int = 2000,
                              end_year: int = 2024) -> pd.DataFrame:
    """Compute historical average and std of cumulative CDD curves."""
    all_curves = []
    for year in range(start_year, end_year + 1):
        cum = compute_cumulative_cdd(cdd_df, year)
        if not cum.empty:
            cum['year'] = year
            all_curves.append(cum[['day_of_season', 'cumulative_cdd', 'year']])

    if not all_curves:
        return pd.DataFrame(columns=['day_of_season', 'mean', 'std', 'upper', 'lower'])

    combined = pd.concat(all_curves, ignore_index=True)
    stats = combined.groupby('day_of_season')['cumulative_cdd'].agg(['mean', 'std']).reset_index()
    stats['upper'] = stats['mean'] + stats['std']
    stats['lower'] = (stats['mean'] - stats['std']).clip(lower=0)
    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 1: CDD Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_cdd_dashboard():
    """Render the CDD Dashboard tab."""
    st.header("🌡️ Cooling Degree Days (CDD) Dashboard")
    st.markdown(
        "Population-weighted CDD by region. "
        "CDD = max(T_mean − 18°C, 0). Cumulative from April 15."
    )

    # Region selector in sidebar
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=list(REGION_MAP.keys()),
        default=['China North', 'China South', 'Japan', 'India'],
    )

    if not selected_regions:
        st.info("Select at least one region from the sidebar.")
        return

    current_year = datetime.now().year
    prev_year = current_year - 1

    # Summary table
    summary_rows = []

    for region in selected_regions:
        with st.spinner(f"Loading data for {region}..."):
            # Load historical
            hist_df = load_historical_temperature(region)
            if hist_df.empty:
                st.warning(f"No historical data found for {region}")
                continue

            # Compute region CDD
            region_cdd = compute_region_cdd(hist_df, region)

            # Load forecast
            fcst_df = load_forecast_temperature(region)
            fcst_cdd = compute_region_cdd(fcst_df, region) if not fcst_df.empty else pd.DataFrame()

            # Combine current year actual + forecast
            current_cdd = region_cdd[region_cdd['date'] >= f'{current_year}-01-01'].copy()
            if not fcst_cdd.empty:
                combined_current = pd.concat([current_cdd, fcst_cdd], ignore_index=True)
                combined_current = combined_current.drop_duplicates(subset='date', keep='first')
                combined_current = combined_current.sort_values('date')
            else:
                combined_current = current_cdd

            # Compute cumulative curves
            cum_current = compute_cumulative_cdd(combined_current, current_year)
            cum_prev = compute_cumulative_cdd(region_cdd, prev_year)
            normal = compute_historical_normal(region_cdd)

            # Build chart
            fig = go.Figure()

            # Historical normal band
            if not normal.empty:
                fig.add_trace(go.Scatter(
                    x=normal['day_of_season'], y=normal['upper'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, name='Upper'
                ))
                fig.add_trace(go.Scatter(
                    x=normal['day_of_season'], y=normal['lower'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.2)',
                    name='Normal ±1σ (2000-2024)'
                ))
                fig.add_trace(go.Scatter(
                    x=normal['day_of_season'], y=normal['mean'],
                    mode='lines', line=dict(color='gray', dash='dot', width=1.5),
                    name='Normal Mean'
                ))

            # Previous year
            if not cum_prev.empty:
                fig.add_trace(go.Scatter(
                    x=cum_prev['day_of_season'], y=cum_prev['cumulative_cdd'],
                    mode='lines', line=dict(color='blue', width=1.5, dash='dash'),
                    name=f'{prev_year}'
                ))

            # Current year (actual part)
            if not cum_current.empty:
                # Split into actual vs forecast
                today = pd.Timestamp.today().normalize()
                actual_mask = cum_current['date'] <= today
                forecast_mask = cum_current['date'] > today

                actual_part = cum_current[actual_mask]
                forecast_part = cum_current[forecast_mask]

                if not actual_part.empty:
                    fig.add_trace(go.Scatter(
                        x=actual_part['day_of_season'], y=actual_part['cumulative_cdd'],
                        mode='lines', line=dict(color='red', width=3),
                        name=f'{current_year} (Actual)'
                    ))

                if not forecast_part.empty:
                    # Connect forecast to actual
                    if not actual_part.empty:
                        connect = pd.concat([actual_part.tail(1), forecast_part])
                    else:
                        connect = forecast_part
                    fig.add_trace(go.Scatter(
                        x=connect['day_of_season'], y=connect['cumulative_cdd'],
                        mode='lines', line=dict(color='red', width=2, dash='dash'),
                        name=f'{current_year} (Forecast)'
                    ))

            fig.update_layout(
                title=f"Cumulative CDD — {region}",
                xaxis_title="Days since April 15",
                yaxis_title="Cumulative CDD (°C·days)",
                height=400,
                margin=dict(l=50, r=20, t=50, b=40),
                legend=dict(x=0.02, y=0.98),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            total_cdd = cum_current['cumulative_cdd'].iloc[-1] if not cum_current.empty else 0
            normal_cdd = normal.loc[
                normal['day_of_season'] == len(cum_current), 'mean'
            ].values[0] if not normal.empty and len(cum_current) > 0 and len(cum_current) <= len(normal) else 0

            summary_rows.append({
                'Region': region,
                'CDD to Date': f"{total_cdd:.1f}",
                'Normal': f"{normal_cdd:.1f}",
                'Anomaly': f"{total_cdd - normal_cdd:+.1f}",
                'Days in Season': len(cum_current),
            })

    # Summary table
    if summary_rows:
        st.subheader("📊 Season Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 2: Temperature Anomaly Map
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_anomaly_map():
    """Render the Temperature Anomaly Map tab."""
    st.header("🗺️ Temperature Anomaly Map")
    st.markdown(
        "Forecast temperature anomaly vs climatology (ERA5 day-of-year average)."
    )

    forecast_day = st.slider("Forecast Day Ahead", min_value=1, max_value=14, value=3)
    target_date = (datetime.now() + timedelta(days=forecast_day)).date()

    st.caption(f"Showing anomaly for: {target_date}")

    with st.spinner("Computing anomalies..."):
        anomalies = compute_city_anomalies(target_date)

    if anomalies.empty:
        st.warning("No forecast data available for the selected date.")
        return

    fig = px.scatter_geo(
        anomalies,
        lat='latitude',
        lon='longitude',
        color='anomaly',
        hover_name='city',
        hover_data={'temperature': ':.1f', 'climatology': ':.1f', 'anomaly': ':.1f'},
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        range_color=[-8, 8],
        size='marker_size',
        title=f"Temperature Anomaly (°C) — {target_date}",
        projection='natural earth',
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
    fig.update_geos(showland=True, landcolor='lightgray', showcountries=True)
    st.plotly_chart(fig, use_container_width=True)

    # Table view
    st.subheader("City Details")
    display_df = anomalies[['city', 'region', 'temperature', 'climatology', 'anomaly']].copy()
    display_df.columns = ['City', 'Region', 'Forecast (°C)', 'Climatology (°C)', 'Anomaly (°C)']
    display_df = display_df.sort_values('Anomaly (°C)', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=3600)
def compute_city_anomalies(target_date) -> pd.DataFrame:
    """Compute forecast anomaly vs climatology for all cities."""
    # Build coordinate filter for ALL cities
    all_cities = list(CITY_LOCATIONS.keys())
    coord_filter = build_coord_filter(all_cities)
    target_date_str = str(target_date)

    # Get forecast for target date
    fcst_query = f"""
    SELECT
        CAST(delivery_start AS DATE) as date,
        AVG(value) as temperature,
        latitude,
        longitude
    FROM dna_prod_silver.meteomatics.temperature_forecast
    WHERE model = 'ecmwf-ens'
      AND curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d'
      AND CAST(delivery_start AS DATE) = '{target_date_str}'
      AND ({coord_filter})
    GROUP BY CAST(delivery_start AS DATE), latitude, longitude
    """

    fcst_df = run_sql_query(fcst_query)
    if fcst_df.empty:
        return pd.DataFrame()

    fcst_df['temperature'] = fcst_df['temperature'].astype(float)
    fcst_df['latitude'] = fcst_df['latitude'].astype(float)
    fcst_df['longitude'] = fcst_df['longitude'].astype(float)

    # Get climatology (day-of-year average from ERA5)
    doy = pd.Timestamp(target_date).day_of_year
    clim_query = f"""
    SELECT
        AVG(value) as climatology,
        latitude,
        longitude
    FROM dna_prod_silver.meteomatics.temperature
    WHERE model = 'ecmwf-era5'
      AND curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d'
      AND DAYOFYEAR(delivery_start) = {doy}
      AND YEAR(delivery_start) BETWEEN 2000 AND 2024
      AND ({coord_filter})
    GROUP BY latitude, longitude
    """

    clim_df = run_sql_query(clim_query)
    if clim_df.empty:
        return pd.DataFrame()

    clim_df['climatology'] = clim_df['climatology'].astype(float)
    clim_df['latitude'] = clim_df['latitude'].astype(float)
    clim_df['longitude'] = clim_df['longitude'].astype(float)

    # Merge forecast and climatology
    merged = fcst_df.merge(clim_df, on=['latitude', 'longitude'], how='inner')
    merged['anomaly'] = merged['temperature'] - merged['climatology']

    # Map to city names and regions
    coord_to_city = {}
    city_to_region = {}
    for region, cities in REGION_MAP.items():
        for city in cities:
            loc = CITY_LOCATIONS[city]
            coord_to_city[(loc['latitude'], loc['longitude'])] = city
            city_to_region[city] = region

    merged['city'] = merged.apply(
        lambda r: coord_to_city.get((r['latitude'], r['longitude']), 'Unknown'), axis=1
    )
    merged['region'] = merged['city'].map(city_to_region)
    merged['marker_size'] = 10  # Uniform marker size

    return merged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 3: City Detail
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_city_detail():
    """Render the City Detail tab."""
    st.header("🏙️ City Detail View")

    city = st.selectbox("Select City", options=sorted(CITY_LOCATIONS.keys()))

    if not city:
        return

    loc = CITY_LOCATIONS[city]
    region = next((r for r, cities in REGION_MAP.items() if city in cities), 'Unknown')

    st.caption(f"Region: {region} | Lat: {loc['latitude']}, Lon: {loc['longitude']} | Pop: {POPULATION[city]:,}")

    with st.spinner(f"Loading data for {city}..."):
        city_data = load_city_data(city)

    if city_data.empty:
        st.warning(f"No data available for {city}")
        return

    # Split historical and forecast
    today = pd.Timestamp.today().normalize()
    hist = city_data[city_data['date'] <= today].copy()
    fcst = city_data[city_data['date'] > today].copy()

    # Temperature time series
    fig_temp = go.Figure()
    if not hist.empty:
        fig_temp.add_trace(go.Scatter(
            x=hist['date'], y=hist['temperature'],
            mode='lines', name='Historical',
            line=dict(color='steelblue', width=1.5)
        ))
    if not fcst.empty:
        # Connect with last historical point
        if not hist.empty:
            connect = pd.concat([hist.tail(1), fcst])
        else:
            connect = fcst
        fig_temp.add_trace(go.Scatter(
            x=connect['date'], y=connect['temperature'],
            mode='lines', name='Forecast (ECMWF-ENS)',
            line=dict(color='red', width=2, dash='dash')
        ))

    # Add base temperature line
    all_dates = city_data['date']
    fig_temp.add_hline(y=BASE_TEMP, line_dash='dot', line_color='green',
                       annotation_text=f"Base ({BASE_TEMP}°C)")

    fig_temp.update_layout(
        title=f"Daily Mean Temperature — {city}",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        height=350, margin=dict(l=50, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # Daily CDD bar chart
    city_data['cdd'] = (city_data['temperature'] - BASE_TEMP).clip(lower=0)
    recent = city_data[city_data['date'] >= today - timedelta(days=30)]

    fig_cdd = go.Figure()
    hist_recent = recent[recent['date'] <= today]
    fcst_recent = recent[recent['date'] > today]

    if not hist_recent.empty:
        fig_cdd.add_trace(go.Bar(
            x=hist_recent['date'], y=hist_recent['cdd'],
            name='Actual CDD', marker_color='steelblue'
        ))
    if not fcst_recent.empty:
        fig_cdd.add_trace(go.Bar(
            x=fcst_recent['date'], y=fcst_recent['cdd'],
            name='Forecast CDD', marker_color='coral'
        ))

    fig_cdd.update_layout(
        title=f"Daily CDD — {city} (Last 30 days + Forecast)",
        xaxis_title="Date", yaxis_title="CDD (°C·days)",
        height=300, margin=dict(l=50, r=20, t=50, b=40),
        barmode='stack'
    )
    st.plotly_chart(fig_cdd, use_container_width=True)

    # Cumulative CDD vs normal
    current_year = datetime.now().year
    city_cdd_df = city_data[['date', 'cdd']].copy()
    cum_current = compute_cumulative_cdd(city_cdd_df, current_year)

    # Get historical for this single city
    hist_full = city_data[city_data['date'] < f'{current_year}-01-01']
    hist_cdd_full = hist_full[['date', 'cdd']].copy() if not hist_full.empty else pd.DataFrame(columns=['date', 'cdd'])
    normal = compute_historical_normal(hist_cdd_full) if not hist_cdd_full.empty else pd.DataFrame()

    fig_cum = go.Figure()
    if not normal.empty:
        fig_cum.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['upper'],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_cum.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['lower'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(128,128,128,0.2)',
            name='Normal ±1σ'
        ))
        fig_cum.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['mean'],
            mode='lines', line=dict(color='gray', dash='dot'),
            name='Normal Mean'
        ))

    if not cum_current.empty:
        fig_cum.add_trace(go.Scatter(
            x=cum_current['day_of_season'], y=cum_current['cumulative_cdd'],
            mode='lines', line=dict(color='red', width=3),
            name=f'{current_year}'
        ))

    fig_cum.update_layout(
        title=f"Cumulative CDD — {city}",
        xaxis_title="Days since April 15",
        yaxis_title="Cumulative CDD",
        height=350, margin=dict(l=50, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_cum, use_container_width=True)


@st.cache_data(ttl=1800)
def load_city_data(city: str) -> pd.DataFrame:
    """Load combined historical + forecast data for a single city."""
    loc = CITY_LOCATIONS[city]
    lat, lon = loc['latitude'], loc['longitude']
    lookback_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    # Historical (last 90 days)
    hist_query = f"""
    SELECT
        CAST(delivery_start AS DATE) as date,
        value as temperature
    FROM dna_prod_silver.meteomatics.temperature
    WHERE model = 'ecmwf-era5'
      AND curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d'
      AND latitude = {lat}
      AND longitude = {lon}
      AND delivery_start >= '{lookback_date}'
    ORDER BY delivery_start
    """

    # Forecast
    fcst_query = f"""
    SELECT
        CAST(delivery_start AS DATE) as date,
        AVG(value) as temperature
    FROM dna_prod_silver.meteomatics.temperature_forecast
    WHERE model = 'ecmwf-ens'
      AND curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d'
      AND latitude = {lat}
      AND longitude = {lon}
      AND delivery_start >= CURRENT_DATE()
    GROUP BY CAST(delivery_start AS DATE)
    ORDER BY date
    """

    hist_df = run_sql_query(hist_query)
    fcst_df = run_sql_query(fcst_query)

    frames = []
    if not hist_df.empty:
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df['temperature'] = hist_df['temperature'].astype(float)
        frames.append(hist_df)
    if not fcst_df.empty:
        fcst_df['date'] = pd.to_datetime(fcst_df['date'])
        fcst_df['temperature'] = fcst_df['temperature'].astype(float)
        frames.append(fcst_df)

    if not frames:
        return pd.DataFrame(columns=['date', 'temperature'])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset='date', keep='first').sort_values('date')
    return combined


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.sidebar.title("⚡ Coal Desk Weather")
    st.sidebar.markdown("LPG/CDD Report")
    st.sidebar.divider()

    tab1, tab2, tab3 = st.tabs([
        "📈 CDD Dashboard",
        "🗺️ Temperature Anomaly Map",
        "🏙️ City Detail",
    ])

    with tab1:
        render_cdd_dashboard()

    with tab2:
        render_anomaly_map()

    with tab3:
        render_city_detail()


if __name__ == "__main__":
    main()
