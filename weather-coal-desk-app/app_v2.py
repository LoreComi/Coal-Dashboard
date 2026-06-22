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
    load_all_historical_cumulative, compute_similar_years,
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

            # Similarity analysis — pre-compute all historical trajectories once
            all_hist_cum = load_all_historical_cumulative(region_cdd)
            sim_years = compute_similar_years(region_cdd, cum_current)

            fig = make_cumulative_cdd_chart(
                region, cum_current, cum_prev, normal, current_year,
                all_historical_cumulative=all_hist_cum,
                similar_years=sim_years,
            )
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
    fig.add_hline(y=BASE_TEMP, line_dash='dot', line_color='#16a34a', annotation_text=f"CDD base ({BASE_TEMP}°C)", annotation_font_color='#16a34a')
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{region} — {model_choice}", font=dict(size=14)), xaxis_title="Date", yaxis_title="°C", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### CITY BREAKDOWN")
    latest = df_region[df_region['date'] == df_region['date'].max()]
    if not latest.empty:
        pivot = latest.pivot_table(index='city', columns='parameter', values='value', aggfunc='mean')
        pivot.columns = [c.replace('_2m_24h','').replace('t_','T_') for c in pivot.columns]
        st.dataframe(pivot.round(1).sort_index(), use_container_width=True)


# ─── Tab 3: Anomaly Map ──────────────────────────────────────────────────────────

# Light matplotlib style matching the app's professional light theme
_MPL_STYLE = {
    'figure.facecolor': '#f8fafc',
    'axes.facecolor':   '#ffffff',
    'text.color':       '#0f172a',
    'axes.labelcolor':  '#374151',
    'xtick.color':      '#6b7280',
    'ytick.color':      '#6b7280',
    'axes.titlecolor':  '#0f172a',
    'axes.edgecolor':   '#e2e8f0',
    'grid.color':       '#e2e8f0',
    'axes.titlesize':   13,
    'font.family':      'sans-serif',
}
_LAND_COLOR   = '#e2e8f0'
_OCEAN_COLOR  = '#dbeafe'
_COAST_COLOR  = '#475569'
_BORDER_COLOR = '#94a3b8'


def _make_map_fig(lons, lats, data_2d, bounds, title, cmap, levels, cbar_label,
                  has_cartopy):
    """Render a single contourf map (cartopy or plain matplotlib) with dark theme."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    with plt.rc_context(_MPL_STYLE):
        if has_cartopy:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig_mpl, ax = plt.subplots(
                1, 1, figsize=(12, 7),
                subplot_kw={'projection': ccrs.PlateCarree()},
            )
            ax.set_facecolor(_OCEAN_COLOR)
            ax.set_extent(
                [bounds['lon_min'], bounds['lon_max'],
                 bounds['lat_min'], bounds['lat_max']],
                crs=ccrs.PlateCarree(),
            )
            ax.add_feature(cfeature.OCEAN, color=_OCEAN_COLOR, zorder=0)
            ax.add_feature(cfeature.LAND,  color=_LAND_COLOR,  zorder=0)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5,
                           edgecolor=_BORDER_COLOR, zorder=2)
            ax.coastlines(linewidth=0.8, color=_COAST_COLOR, zorder=2)
            cf = ax.contourf(
                lons, lats, data_2d, levels=levels,
                cmap=cmap, extend='both', transform=ccrs.PlateCarree(), zorder=1,
            )
            gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                              color='#e2e8f0', alpha=0.9)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'color': '#6b7280', 'size': 8}
            gl.ylabel_style = {'color': '#6b7280', 'size': 8}
        else:
            fig_mpl, ax = plt.subplots(1, 1, figsize=(12, 7))
            cf = ax.contourf(lons, lats, data_2d, levels=levels, cmap=cmap, extend='both')
            ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
            ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
            ax.set_xlabel('Longitude', color='#7d9ab8')
            ax.set_ylabel('Latitude', color='#7d9ab8')

        cbar = fig_mpl.colorbar(cf, ax=ax, orientation='horizontal',
                                pad=0.06, fraction=0.046, label=cbar_label)
        cbar.ax.xaxis.label.set_color('#374151')
        cbar.ax.tick_params(colors='#6b7280')
        cbar.outline.set_edgecolor('#e2e8f0')
        ax.set_title(title, pad=10)
        plt.tight_layout()
    return fig_mpl


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

    bounds = MAP_REGIONS[map_region]
    today_d = datetime.now().date()

    import matplotlib
    matplotlib.use('Agg')
    try:
        import cartopy.crs as ccrs      # noqa: F401
        import cartopy.feature as cfeature  # noqa: F401
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    periods = [
        (today_d + timedelta(days=i * 5),
         today_d + timedelta(days=(i + 1) * 5 - 1))
        for i in range(3)
    ]

    # ── Temperature deviation maps ────────────────────────────────────────────
    for p_start, p_end in periods:
        with st.spinner(f"Loading T anomaly  {p_start:%d %b} – {p_end:%d %b %Y}…"):
            try:
                period_df = load_gridded_anomalies_multiday(map_region, p_start, p_end)
            except Exception as e:
                st.error(f"Error: {e}")
                continue
        if period_df.empty:
            st.warning(f"No temperature data for {p_start} – {p_end}")
            continue

        lats = np.sort(period_df['latitude'].unique())
        lons = np.sort(period_df['longitude'].unique())
        pivot = period_df.pivot_table(index='latitude', columns='longitude', values='anomaly')
        data_2d = pivot.reindex(index=lats, columns=lons).values

        fig_mpl = _make_map_fig(
            lons, lats, data_2d, bounds,
            title=f"Temperature Deviation from Climatology\n"
                  f"{map_region}  ·  {p_start:%d %b} – {p_end:%d %b %Y}",
            cmap='RdBu_r',
            levels=np.linspace(-10, 10, 21),
            cbar_label='T2m Anomaly (°C)',
            has_cartopy=has_cartopy,
        )
        st.pyplot(fig_mpl)
        import matplotlib.pyplot as plt
        plt.close(fig_mpl)

    # ── Precipitation deviation maps ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### PRECIPITATION DEVIATION")

    for p_start, p_end in periods:
        with st.spinner(f"Loading precip deviation  {p_start:%d %b} – {p_end:%d %b %Y}…"):
            try:
                precip_df = load_gridded_precip_deviation(map_region, p_start, p_end)
            except Exception as e:
                st.error(f"Error: {e}")
                continue
        if precip_df.empty:
            st.warning(f"No precipitation data for {p_start} – {p_end}")
            continue

        lats = np.sort(precip_df['latitude'].unique())
        lons = np.sort(precip_df['longitude'].unique())
        pivot = precip_df.pivot_table(index='latitude', columns='longitude', values='anomaly')
        data_2d = pivot.reindex(index=lats, columns=lons).values

        fig_mpl = _make_map_fig(
            lons, lats, data_2d, bounds,
            title=f"Precipitation Deviation from Climatology\n"
                  f"{map_region}  ·  {p_start:%d %b} – {p_end:%d %b %Y}",
            cmap='BrBG',
            levels=np.linspace(-8, 8, 17),
            cbar_label='Precip Deviation (mm/day)',
            has_cartopy=has_cartopy,
        )
        st.pyplot(fig_mpl)
        import matplotlib.pyplot as plt
        plt.close(fig_mpl)


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
            fig_temp.add_trace(go.Scatter(x=tmin['date'], y=tmin['value'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(234,88,12,0.10)', name='T_min–T_max'))
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
