"""Coal Desk Weather / CDD Report — Trader Edition
========================================================
Built for LPG & coal traders. Population-weighted CDD across
57 cities, 16 regions. Data from Databricks Unity Catalog.

Module layout
-------------
    main.py        ← this file (UI / orchestration)
    _config.py     ← cities, regions, populations, constants
    _data.py       ← SQL & caching layer (REST API)
    _charts.py     ← Plotly figure builders
    _style.py      ← CSS + Plotly dark-navy theme
"""
from __future__ import annotations

import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    REGION_MAP, CITY_LOCATIONS, POPULATION, CITY_TO_REGION,
    BASE_TEMP, DEFAULT_REGIONS,
)
from _data import (
    load_historical, load_forecast, load_city_timeseries, load_anomalies,
    compute_region_cdd, compute_cumulative, compute_normal,
)
from _charts import (
    make_cumulative_cdd_chart, make_temperature_chart,
    make_daily_cdd_bars, make_anomaly_map,
)
from _style import CUSTOM_CSS


# ─── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coal Desk CDD — Trader Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── KPI card helper ────────────────────────────────────────────────────────────

def kpi_card(label: str, value, unit: str, delta=None, card_class: str = "") -> str:
    val_str = f"{value:+.1f}" if isinstance(value, (int, float)) and not np.isnan(value) else "N/A"
    delta_html = ""
    if delta is not None and not np.isnan(delta):
        d_class = "kpi-delta-up" if delta > 0 else ("kpi-delta-down" if delta < 0 else "kpi-delta-flat")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "●")
        delta_html = f'<div class="kpi-delta {d_class}">{arrow} {abs(delta):.1f} vs normal</div>'
    return f"""
    <div class="kpi-card {card_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val_str} {unit}</div>
        {delta_html}
    </div>
    """


# ─── Tab 1: CDD Dashboard ────────────────────────────────────────────────────────

def render_cdd_dashboard():
    st.markdown("#### COOLING DEGREE DAYS")
    st.caption("Population-weighted CDD by region. CDD = max(T_mean − 18°C, 0). Cumulative from April 15.")

    selected = st.multiselect(
        "Regions", options=list(REGION_MAP.keys()), default=DEFAULT_REGIONS,
        label_visibility="collapsed",
    )
    if not selected:
        st.info("Select at least one region.")
        return

    current_year = datetime.now().year
    prev_year = current_year - 1
    summary_rows = []

    # Render 2 charts per row
    cols = st.columns(2)
    for idx, region in enumerate(selected):
        try:
            hist_df = load_historical(region)
            if hist_df.empty:
                continue
            region_cdd = compute_region_cdd(hist_df, region)

            fcst_df = load_forecast(region)
            fcst_cdd = compute_region_cdd(fcst_df, region) if not fcst_df.empty else pd.DataFrame(columns=['date', 'cdd'])

            # Combine actual + forecast for current year
            current = region_cdd[region_cdd['date'] >= f'{current_year}-01-01'].copy()
            if not fcst_cdd.empty:
                combined = pd.concat([current, fcst_cdd]).drop_duplicates('date', keep='first').sort_values('date')
            else:
                combined = current

            cum_current = compute_cumulative(combined, current_year)
            cum_prev = compute_cumulative(region_cdd, prev_year)
            normal = compute_normal(region_cdd)

            fig = make_cumulative_cdd_chart(region, cum_current, cum_prev, normal, current_year)

            with cols[idx % 2]:
                st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            total = cum_current['cumulative_cdd'].iloc[-1] if not cum_current.empty else 0
            n_days = len(cum_current)
            normal_val = normal.loc[normal['day_of_season'] == n_days, 'mean'].values[0] if (
                not normal.empty and n_days > 0 and n_days <= len(normal)) else 0
            summary_rows.append({
                'Region': region,
                'CDD to Date': f"{total:.0f}",
                'Normal': f"{normal_val:.0f}",
                'Anomaly': f"{total - normal_val:+.0f}",
                'Days': n_days,
            })
        except Exception as e:
            with cols[idx % 2]:
                st.error(f"{region}: {e}")

    # KPI summary
    if summary_rows:
        st.markdown("---")
        st.markdown("#### SEASON SUMMARY")
        kpi_cols = st.columns(min(len(summary_rows), 6))
        for i, row in enumerate(summary_rows[:6]):
            anomaly = float(row['Anomaly'])
            cls = "kpi-card-warm" if anomaly > 0 else "kpi-card-cool"
            with kpi_cols[i]:
                st.markdown(kpi_card(row['Region'], anomaly, "°C·d vs normal", card_class=cls), unsafe_allow_html=True)

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ─── Tab 2: Anomaly Map ──────────────────────────────────────────────────────────

def render_anomaly_map():
    st.markdown("#### TEMPERATURE ANOMALY MAP")
    st.caption("Forecast vs ERA5 climatology (2000–2024 day-of-year average).")

    forecast_day = st.slider("Forecast day ahead", 1, 14, 3, key="anomaly_slider")
    target_date = (datetime.now() + timedelta(days=forecast_day)).date()
    st.caption(f"Target: **{target_date}**")

    try:
        anomalies = load_anomalies(target_date)
    except Exception as e:
        st.error(f"Error loading anomalies: {e}")
        return

    if anomalies.empty:
        st.warning("No forecast data for the selected date.")
        return

    fig = make_anomaly_map(anomalies, target_date)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("#### CITY BREAKDOWN")
    disp = anomalies[['city', 'region', 'temperature', 'climatology', 'anomaly']].copy()
    disp.columns = ['City', 'Region', 'Forecast °C', 'Clim °C', 'Anomaly °C']
    for c in ['Forecast °C', 'Clim °C', 'Anomaly °C']:
        disp[c] = disp[c].astype(float).round(1)
    disp = disp.sort_values('Anomaly °C', ascending=False)
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ─── Tab 3: City Detail ──────────────────────────────────────────────────────────

def render_city_detail():
    st.markdown("#### CITY DETAIL")

    city = st.selectbox("City", sorted(CITY_LOCATIONS.keys()), label_visibility="collapsed")
    if not city:
        return

    loc = CITY_LOCATIONS[city]
    region = CITY_TO_REGION.get(city, '?')
    st.caption(f"Region: **{region}** | Lat {loc['latitude']}°, Lon {loc['longitude']}° | Pop: {POPULATION[city]:,}")

    try:
        data = load_city_timeseries(city)
    except Exception as e:
        st.error(f"Error: {e}")
        return

    if data.empty:
        st.warning(f"No data for {city}.")
        return

    # Temperature chart
    fig_temp = make_temperature_chart(city, data)
    st.plotly_chart(fig_temp, use_container_width=True)

    # Daily CDD bars
    fig_cdd = make_daily_cdd_bars(city, data)
    st.plotly_chart(fig_cdd, use_container_width=True)

    # Cumulative CDD
    current_year = datetime.now().year
    data_cdd = data.copy()
    data_cdd['cdd'] = (data_cdd['temperature'] - BASE_TEMP).clip(lower=0)
    cum = compute_cumulative(data_cdd, current_year)

    # Simple single-city normal (from historical if available in timeseries)
    if not cum.empty:
        from _charts import make_cumulative_cdd_chart
        # For city detail we just show current year curve (no historical norm)
        fig_cum = make_cumulative_cdd_chart(
            city, cum, pd.DataFrame(), pd.DataFrame(), current_year
        )
        st.plotly_chart(fig_cum, use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("Coal Desk CDD")

    tab1, tab2, tab3 = st.tabs([
        "CDD Dashboard",
        "Anomaly Map",
        "City Detail",
    ])

    with tab1:
        render_cdd_dashboard()
    with tab2:
        render_anomaly_map()
    with tab3:
        render_city_detail()


if __name__ == "__main__":
    main()
