"""Coal Desk Weather / CDD Report — Trader Edition (v2)

Full feature set using pre-computed tables from the ingestion pipeline.
Includes t_min/t_max/t_mean, 44-day vareps forecast, and CDD analytics.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from _config import (
    REGION_MAP, CITY_LOCATIONS, POPULATION, CITY_TO_REGION,
    BASE_TEMP, DEFAULT_REGIONS,
)
from _data_v2 import (
    load_historical, load_forecast, load_city_timeseries, load_anomalies,
    compute_region_cdd, compute_cumulative, compute_normal,
    load_precomputed_cdd, load_precomputed_historical, load_precomputed_forecasts,
    load_current_year_cdd, load_gridded_anomalies, load_gridded_anomalies_multiday,
    load_gridded_precip_deviation, MAP_REGIONS,
)
from _charts import (
    make_cumulative_cdd_chart, make_temperature_chart,
    make_daily_cdd_bars, make_anomaly_map,
)
from _style import CUSTOM_CSS, PLOTLY_LAYOUT


st.set_page_config(
    page_title="Coal Desk CDD — Trader Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def kpi_card(label, value, unit, delta=None, card_class=""):
    val_str = f"{value:+.1f}" if isinstance(value, (int, float)) and not np.isnan(value) else "N/A"
    delta_html = ""
    if delta is not None and not np.isnan(delta):
        d_class = "kpi-delta-up" if delta > 0 else ("kpi-delta-down" if delta < 0 else "kpi-delta-flat")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "●")
        delta_html = f'<div class="kpi-delta {d_class}">{arrow} {abs(delta):.1f} vs normal</div>'
    return f'<div class="kpi-card {card_class}"><div class="kpi-label">{label}</div><div class="kpi-value">{val_str} {unit}</div>{delta_html}</div>'


# ─── Tab 1: CDD Dashboard ────────────────────────────────────────────────────────
def render_cdd_dashboard():
    st.markdown("#### COOLING DEGREE DAYS")
    st.caption("Population-weighted CDD. Includes ecmwf-ens (14d) + ecmwf-vareps (44d extended).")

    selected = st.multiselect("Regions", list(REGION_MAP.keys()), DEFAULT_REGIONS, label_visibility="collapsed")
    if not selected:
        st.info("Select at least one region.")
        return

    current_year = datetime.now().year
    summary_rows = []

    try:
        precomp_hist = load_precomputed_historical()
    except Exception:
        precomp_hist = pd.DataFrame()

    cols = st.columns(2)
    for idx, region in enumerate(selected):
        try:
            # Historical CDD for normal/prev year (from precomputed table: 2000-2024)
            if not precomp_hist.empty:
                region_cdd = precomp_hist[precomp_hist['region'] == region][['date', 'cdd']].copy()
            else:
                hist_df = load_historical(region)
                region_cdd = compute_region_cdd(hist_df, region) if not hist_df.empty else pd.DataFrame(columns=['date', 'cdd'])

            # Current year: ERA5 actuals + forecast from production tables
            combined = load_current_year_cdd(region)

            cum_current = compute_cumulative(combined, current_year)
            cum_prev = compute_cumulative(region_cdd, current_year - 1)
            normal = compute_normal(region_cdd)
            fig = make_cumulative_cdd_chart(region, cum_current, cum_prev, normal, current_year)
            with cols[idx % 2]:
                st.plotly_chart(fig, use_container_width=True)

            total = cum_current['cumulative_cdd'].iloc[-1] if not cum_current.empty else 0
            n_days = len(cum_current)
            nv = normal.loc[normal['day_of_season'] == n_days, 'mean'].values
            normal_val = nv[0] if len(nv) else 0
            summary_rows.append({'Region': region, 'CDD': f"{total:.0f}", 'Normal': f"{normal_val:.0f}", 'Anomaly': f"{total-normal_val:+.0f}", 'Days': n_days})
        except Exception as e:
            with cols[idx % 2]:
                st.error(f"{region}: {e}")

    if summary_rows:
        st.markdown("---")
        st.markdown("#### SEASON SUMMARY")
        kpi_cols = st.columns(min(len(summary_rows), 6))
        for i, row in enumerate(summary_rows[:6]):
            a = float(row['Anomaly'])
            with kpi_cols[i]:
                st.markdown(kpi_card(row['Region'], a, "°C·d", card_class="kpi-card-warm" if a > 0 else "kpi-card-cool"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ─── Tab 2: Forecast Overview ───────────────────────────────────────────────────
def render_forecast_overview():
    st.markdown("#### FORECAST OVERVIEW")
    st.caption("City-level t_min / t_max / t_mean — ecmwf-ens (14d) and ecmwf-vareps (44d).")

    try:
        forecasts = load_precomputed_forecasts()
    except Exception as e:
        st.error(f"Error: {e}")
        return
    if forecasts.empty:
        st.warning("No forecast data. Run the ingestion pipeline first.")
        return

    model_choice = st.radio("Model", ['ecmwf-ens (14d)', 'ecmwf-vareps (44d)'], horizontal=True)
    model_filter = 'ecmwf-ens' if '14d' in model_choice else 'ecmwf-vareps'
    df = forecasts[forecasts['model'] == model_filter]

    region = st.selectbox("Region", list(REGION_MAP.keys()), key="fcst_region")
    df_region = df[df['city'].isin(REGION_MAP[region])]
    if df_region.empty:
        st.warning(f"No {model_filter} data for {region}.")
        return

    fig = go.Figure()
    for param, color, name in [('t_max_2m_24h','#f87171','T_max'), ('t_mean_2m_24h','#e4eeff','T_mean'), ('t_min_2m_24h','#60a5fa','T_min')]:
        p = df_region[df_region['parameter'] == param]
        if not p.empty:
            avg = p.groupby('date')['value'].mean().reset_index()
            fig.add_trace(go.Scatter(x=avg['date'], y=avg['value'], mode='lines', name=name, line=dict(color=color, width=2 if 'mean' in param else 1.5)))
    fig.add_hline(y=BASE_TEMP, line_dash='dot', line_color='#22d3ee', annotation_text=f"CDD base ({BASE_TEMP}°C)", annotation_font_color='#22d3ee')
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{region} — {model_choice}", font=dict(size=14)), xaxis_title="Date", yaxis_title="°C", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### CITY BREAKDOWN")
    latest = df_region[df_region['date'] == df_region['date'].max()]
    if not latest.empty:
        pivot = latest.pivot_table(index='city', columns='parameter', values='value', aggfunc='mean')
        pivot.columns = [c.replace('_2m_24h','').replace('t_','T_') for c in pivot.columns]
        st.dataframe(pivot.round(1).sort_index(), use_container_width=True)


# ─── Tab 3: Anomaly Map ──────────────────────────────────────────────────────────
def render_anomaly_map():
    st.markdown("#### TEMPERATURE ANOMALY MAP")
    st.caption("Gridded forecast deviation from ERA5 climatology (2000–2024). Data: ECMWF-ENS.")

    col1, col2 = st.columns([1, 2])
    with col1:
        forecast_day = st.slider("Forecast day ahead", 1, 14, 3, key="anomaly_slider")
    with col2:
        map_region = st.selectbox("Region", list(MAP_REGIONS.keys()), key="map_region")

    target_date = (datetime.now() + timedelta(days=forecast_day)).date()
    st.caption(f"Target: **{target_date}** | Region: **{map_region}**")

    with st.spinner(f"Loading gridded anomaly map for {map_region}..."):
        try:
            grid_df = load_gridded_anomalies(map_region, target_date)
        except Exception as e:
            st.error(f"Error loading map data: {e}")
            return

    if grid_df.empty:
        st.warning("No forecast data available for the selected date and region.")
        return

    bounds = MAP_REGIONS[map_region]
    today_d = datetime.now().date()

    # Generate 5-day period maps with cartopy (matching notebook output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    periods = []
    for i in range(3):
        p_start = today_d + timedelta(days=i * 5)
        p_end = today_d + timedelta(days=(i + 1) * 5 - 1)
        periods.append((p_start, p_end))

    for p_start, p_end in periods:
        with st.spinner(f"Generating map for {p_start} to {p_end}..."):
            period_df = load_gridded_anomalies_multiday(map_region, p_start, p_end)
        if period_df.empty:
            st.warning(f"No data for {p_start} to {p_end}")
            continue

        lats = np.sort(period_df['latitude'].unique())
        lons = np.sort(period_df['longitude'].unique())
        grid_pivot = period_df.pivot_table(index='latitude', columns='longitude', values='anomaly')
        grid_pivot = grid_pivot.reindex(index=lats, columns=lons)
        data_2d = grid_pivot.values

        if has_cartopy:
            fig_mpl, ax = plt.subplots(1, 1, figsize=(12, 7),
                subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([bounds['lon_min'], bounds['lon_max'],
                           bounds['lat_min'], bounds['lat_max']], crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, color='#e6f2ff')
            ax.add_feature(cfeature.LAND, color='#f5f5f5')
            levels = np.linspace(-10, 10, 21)
            cf = ax.contourf(lons, lats, data_2d, levels=levels,
                             cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08,
                         fraction=0.046, label='T2m Anomaly (deg C)')
        else:
            fig_mpl, ax = plt.subplots(1, 1, figsize=(12, 7))
            levels = np.linspace(-10, 10, 21)
            cf = ax.contourf(lons, lats, data_2d, levels=levels,
                             cmap='RdBu_r', extend='both')
            ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
            ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
            plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08,
                         fraction=0.046, label='T2m Anomaly (deg C)')

        ax.set_title(f"Temperature Deviation from Climatology\n"
                     f"{map_region} - {p_start.strftime('%d %b')} to {p_end.strftime('%d %b %Y')}",
                     fontsize=13)
        plt.tight_layout()
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)

    # --- Precipitation Deviation Maps ---
    st.markdown("---")
    st.markdown("#### PRECIPITATION DEVIATION")

    for p_start, p_end in periods:
        with st.spinner(f"Generating precip map for {p_start} to {p_end}..."):
            precip_df = load_gridded_precip_deviation(map_region, p_start, p_end)
        if precip_df.empty:
            st.warning(f"No precipitation data for {p_start} to {p_end}")
            continue

        lats = np.sort(precip_df['latitude'].unique())
        lons = np.sort(precip_df['longitude'].unique())
        grid_pivot = precip_df.pivot_table(index='latitude', columns='longitude', values='anomaly')
        grid_pivot = grid_pivot.reindex(index=lats, columns=lons)
        data_2d = grid_pivot.values

        if has_cartopy:
            fig_mpl, ax = plt.subplots(1, 1, figsize=(12, 7),
                subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([bounds['lon_min'], bounds['lon_max'],
                           bounds['lat_min'], bounds['lat_max']], crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, color='#e6f2ff')
            ax.add_feature(cfeature.LAND, color='#f5f5f5')
            levels = np.linspace(-8, 8, 17)
            cf = ax.contourf(lons, lats, data_2d, levels=levels,
                             cmap='BrBG', extend='both', transform=ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08,
                         fraction=0.046, label='Precip Deviation (mm/day)')
        else:
            fig_mpl, ax = plt.subplots(1, 1, figsize=(12, 7))
            levels = np.linspace(-8, 8, 17)
            cf = ax.contourf(lons, lats, data_2d, levels=levels,
                             cmap='BrBG', extend='both')
            ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
            ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
            plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08,
                         fraction=0.046, label='Precip Deviation (mm/day)')

        ax.set_title(f"Precipitation Deviation from Climatology\n"
                     f"{map_region} - {p_start.strftime('%d %b')} to {p_end.strftime('%d %b %Y')}",
                     fontsize=13)
        plt.tight_layout()
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)

    # Legacy single-day map (unreachable, kept for reference)
    import plotly.express as px
    fig = px.scatter_geo(
        grid_df,
        lat='latitude',
        lon='longitude',
        color='anomaly',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        range_color=[-8, 8],
        title=f"Temperature Anomaly (°C) — {map_region} — {target_date}",
        projection='natural earth',
    )
    bounds = MAP_REGIONS[map_region]
    fig.update_traces(marker=dict(size=6, symbol='square', line=dict(width=0)))
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0c1828',
        font=dict(family='Inter, sans-serif', color='#e4eeff'),
        geo=dict(
            bgcolor='#0c1828',
            landcolor='#1a2942',
            showland=True,
            showcountries=True,
            countrycolor='rgba(255,255,255,0.2)',
            coastlinecolor='rgba(255,255,255,0.3)',
            showocean=True,
            oceancolor='#07111f',
            showlakes=False,
            lonaxis=dict(range=[bounds['lon_min'], bounds['lon_max']]),
            lataxis=dict(range=[bounds['lat_min'], bounds['lat_max']]),
        ),
        coloraxis_colorbar=dict(title="°C", len=0.6),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    st.markdown(f"**Grid points:** {len(grid_df)} | "
                f"**Mean anomaly:** {grid_df['anomaly'].mean():+.1f}°C | "
                f"**Max:** {grid_df['anomaly'].max():+.1f}°C | "
                f"**Min:** {grid_df['anomaly'].min():+.1f}°C")


# ─── Tab 4: City Detail ──────────────────────────────────────────────────────────
def render_city_detail():
    st.markdown("#### CITY DETAIL")
    city = st.selectbox("City", sorted(CITY_LOCATIONS.keys()), label_visibility="collapsed")
    if not city:
        return
    loc = CITY_LOCATIONS[city]
    region = CITY_TO_REGION.get(city, '?')
    st.caption(f"Region: **{region}** | Lat {loc['latitude']}°, Lon {loc['longitude']}° | Pop: {POPULATION[city]:,}")

    try:
        forecasts = load_precomputed_forecasts()
        city_fcst = forecasts[forecasts['city'] == city] if not forecasts.empty else pd.DataFrame()
    except Exception:
        city_fcst = pd.DataFrame()

    try:
        data = load_city_timeseries(city)
    except Exception as e:
        st.error(f"Error: {e}")
        return
    if data.empty and city_fcst.empty:
        st.warning(f"No data for {city}.")
        return

    # Temperature with t_min/t_max band
    fig_temp = make_temperature_chart(city, data) if not data.empty else go.Figure()
    if not city_fcst.empty:
        tmin = city_fcst[city_fcst['parameter'] == 't_min_2m_24h'].sort_values('date')
        tmax = city_fcst[city_fcst['parameter'] == 't_max_2m_24h'].sort_values('date')
        if not tmin.empty and not tmax.empty:
            fig_temp.add_trace(go.Scatter(x=tmax['date'], y=tmax['value'], mode='lines', line=dict(width=0), showlegend=False))
            fig_temp.add_trace(go.Scatter(x=tmin['date'], y=tmin['value'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(251,146,60,0.15)', name='T_min–T_max'))
    st.plotly_chart(fig_temp, use_container_width=True)

    if not data.empty:
        fig_cdd = make_daily_cdd_bars(city, data)
        st.plotly_chart(fig_cdd, use_container_width=True)

        # Cumulative CDD with forecast extension
        data_cdd = data.copy()
        data_cdd['cdd'] = (data_cdd['temperature'] - BASE_TEMP).clip(lower=0)
        if not city_fcst.empty:
            tmean_f = city_fcst[city_fcst['parameter'] == 't_mean_2m_24h'][['date','value']].rename(columns={'value':'temperature'})
            tmean_f['cdd'] = (tmean_f['temperature'] - BASE_TEMP).clip(lower=0)
            data_cdd = pd.concat([data_cdd[['date','cdd']], tmean_f[['date','cdd']]]).drop_duplicates('date', keep='first').sort_values('date')
        cum = compute_cumulative(data_cdd, datetime.now().year)
        if not cum.empty:
            fig_cum = make_cumulative_cdd_chart(city, cum, pd.DataFrame(), pd.DataFrame(), datetime.now().year)
            st.plotly_chart(fig_cum, use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("Coal Desk CDD")
    tab1, tab2, tab3, tab4 = st.tabs(["CDD Dashboard", "Forecast Overview", "Anomaly Map", "City Detail"])
    with tab1:
        render_cdd_dashboard()
    with tab2:
        render_forecast_overview()
    with tab3:
        render_anomaly_map()
    with tab4:
        render_city_detail()


if __name__ == "__main__":
    main()
