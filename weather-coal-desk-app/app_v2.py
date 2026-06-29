"""Coal Desk Weather / CDD Report — Trader Edition (v2)

Full feature set using pre-computed tables from the ingestion pipeline.
Includes t_min/t_max/t_mean, 44-day vareps forecast, and CDD analytics.
"""
from __future__ import annotations

import traceback
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
    load_watershed_precip, load_gatun_lake_levels, load_hurricane_data,
    THREE_GORGES_DAM,
    load_current_year_cdd_bulk, load_forecast_spread_bulk, compute_ensemble_spread,
    compute_daily_cdd_climatology_v2, compute_temperature_climatology_simple,
)
from _charts import (
    make_cumulative_cdd_chart, make_temperature_chart,
    make_daily_cdd_bars, make_anomaly_map,
    make_forecast_temperature_chart, make_forecast_cdd_deviation_chart,
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

    # ── Historical CDD (2000-2024): one query for all regions ─────────────────
    try:
        precomp_hist = load_precomputed_historical()
    except Exception:
        precomp_hist = pd.DataFrame()

    # ── Current year: ERA5 actuals + ENS forecast, all regions in two queries ─
    try:
        current_year_bulk = load_current_year_cdd_bulk(tuple(sorted(selected)))
    except Exception:
        current_year_bulk = pd.DataFrame(columns=['date', 'cdd', 'region'])

    # ── Ensemble spread: one bulk query for all selected regions ──────────────
    try:
        spread_bulk = load_forecast_spread_bulk(tuple(sorted(selected)))
    except Exception:
        spread_bulk = pd.DataFrame()

    cols = st.columns(2)
    for idx, region in enumerate(selected):
        try:
            # Historical CDD (2000-2024) for normal / prev year / similar years
            if not precomp_hist.empty and region in precomp_hist['region'].values:
                region_cdd = precomp_hist[precomp_hist['region'] == region][['date', 'cdd']].sort_values('date').reset_index(drop=True)
            else:
                hist_df = load_historical(region)
                region_cdd = compute_region_cdd(hist_df, region) if not hist_df.empty else pd.DataFrame(columns=['date', 'cdd'])

            # Current year: use bulk result (ERA5 actuals + ENS gap-fill), fallback per-region
            if not current_year_bulk.empty and region in current_year_bulk['region'].values:
                combined = current_year_bulk[current_year_bulk['region'] == region][['date', 'cdd']].sort_values('date').reset_index(drop=True)
            else:
                combined = load_current_year_cdd(region)

            cum_current = compute_cumulative(combined, current_year)
            cum_prev = compute_cumulative(region_cdd, current_year - 1)
            normal = compute_normal(region_cdd)

            all_hist_cum = load_all_historical_cumulative(region_cdd)
            sim_years = compute_similar_years(region_cdd, cum_current)

            # Ensemble uncertainty band on the forecast portion
            ensemble_spread = compute_ensemble_spread(spread_bulk, region, cum_current)

            fig = make_cumulative_cdd_chart(
                region, cum_current, cum_prev, normal, current_year,
                all_historical_cumulative=all_hist_cum,
                similar_years=sim_years,
                ensemble_spread=ensemble_spread,
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
                  has_cartopy, figsize=(12, 7), extra_draw=None):
    """Render a single contourf map (cartopy or plain matplotlib)."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_MPL_STYLE):
        if has_cartopy:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig_mpl, ax = plt.subplots(
                1, 1, figsize=figsize,
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
            gl.xlabel_style = {'color': '#6b7280', 'size': 7}
            gl.ylabel_style = {'color': '#6b7280', 'size': 7}
        else:
            fig_mpl, ax = plt.subplots(1, 1, figsize=figsize)
            cf = ax.contourf(lons, lats, data_2d, levels=levels, cmap=cmap, extend='both')
            ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
            ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
            ax.set_xlabel('Longitude', color='#6b7280')
            ax.set_ylabel('Latitude', color='#6b7280')

        if extra_draw is not None:
            extra_draw(ax, has_cartopy)

        cbar = fig_mpl.colorbar(cf, ax=ax, orientation='horizontal',
                                pad=0.06, fraction=0.046, label=cbar_label)
        cbar.ax.xaxis.label.set_color('#374151')
        cbar.ax.tick_params(labelsize=7, colors='#6b7280')
        cbar.outline.set_edgecolor('#e2e8f0')
        ax.set_title(title, pad=8, fontsize=10)
        plt.tight_layout()
    return fig_mpl


def _load_geojson_ring(filename, step=10):
    """Load outer ring of first GeoJSON feature, subsampled every `step` vertices."""
    import json, os
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            g = json.load(f)
        coords = g['features'][0]['geometry']['coordinates'][0]
        return [[c[0], c[1]] for c in coords[::step]]
    except Exception:
        return None


def _draw_three_gorges(ax, has_cartopy):
    """Overlay Three Gorges catchment + full Yangtze basin on a map axis.

    Mirrors notebook: catchment = black solid linewidth 1.8 (zorder 5),
    Yangtze basin = darkblue solid linewidth 1.6 (zorder 4), dam = red star.
    """
    import matplotlib.patches as mpatches
    dam_lon, dam_lat = THREE_GORGES_DAM

    # Subsample: 12 510 → ~1 250 pts for catchment, 17 264 → ~1 150 pts for Yangtze
    tg_ring = _load_geojson_ring('three_gorges_catchment.geojson', step=10)
    yz_ring = _load_geojson_ring('yangtze_basin_precise.geojson',  step=15)

    def _add_patch(ring, edgecolor, linewidth, zorder, transform=None):
        kw = dict(closed=True, linewidth=linewidth, edgecolor=edgecolor,
                  facecolor='none', linestyle='-', alpha=0.9, zorder=zorder)
        if transform is not None:
            kw['transform'] = transform
        ax.add_patch(mpatches.Polygon(ring, **kw))

    if has_cartopy:
        import cartopy.crs as ccrs
        tr = ccrs.PlateCarree()
        if yz_ring:
            _add_patch(yz_ring, 'darkblue', 1.6, 4, tr)
        if tg_ring:
            _add_patch(tg_ring, 'black', 1.8, 5, tr)
        ax.plot(dam_lon, dam_lat, marker='*', color='red', markersize=12,
                transform=tr, zorder=6, linestyle='None')
        ax.text(dam_lon + 0.4, dam_lat + 0.6, 'Three Gorges Dam',
                fontsize=6, fontweight='bold', color='red',
                transform=tr, zorder=6)
    else:
        if yz_ring:
            _add_patch(yz_ring, 'darkblue', 1.6, 4)
        if tg_ring:
            _add_patch(tg_ring, 'black', 1.8, 5)
        ax.plot(dam_lon, dam_lat, marker='*', color='red', markersize=12,
                zorder=6, linestyle='None')
        ax.text(dam_lon + 0.4, dam_lat + 0.6, 'Three Gorges Dam',
                fontsize=6, fontweight='bold', color='red', zorder=6)


def _render_watershed_charts(label: str, hist_df: pd.DataFrame,
                              fcst_df: pd.DataFrame, clim_df: pd.DataFrame):
    """Two-chart layout: daily precipitation vs climatology + cumulative departure."""
    if hist_df.empty and fcst_df.empty:
        st.warning(f"No data available for {label}.")
        return

    # Attach day-of-year and merge climatology
    def _add_clim(df):
        if df.empty:
            return df
        d = df.copy()
        d['doy'] = d['date'].dt.day_of_year
        if not clim_df.empty:
            d = d.merge(clim_df[['doy', 'mean_precip', 'std_precip']], on='doy', how='left')
        else:
            d['mean_precip'] = 0.0
            d['std_precip']  = 0.0
        return d

    obs  = _add_clim(hist_df)
    fcst = _add_clim(fcst_df)

    ch1, ch2 = st.columns(2)

    # ── Chart 1: daily precipitation bars with climatological envelope ────────
    with ch1:
        fig1 = go.Figure()

        if not obs.empty and 'mean_precip' in obs.columns:
            fig1.add_trace(go.Scatter(
                x=obs['date'],
                y=(obs['mean_precip'] + obs['std_precip']).clip(lower=0),
                mode='lines', line=dict(width=0), showlegend=False,
            ))
            fig1.add_trace(go.Scatter(
                x=obs['date'],
                y=(obs['mean_precip'] - obs['std_precip']).clip(lower=0),
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(156,163,175,0.25)',
                name='Clim ±1σ',
            ))
            fig1.add_trace(go.Scatter(
                x=obs['date'], y=obs['mean_precip'],
                mode='lines', line=dict(color='#9ca3af', dash='dash', width=1.2),
                name='Clim mean',
            ))
            anom_col = obs['precipitation'] - obs['mean_precip']
            bar_colors = ['#1d4ed8' if v >= 0 else '#dc2626' for v in anom_col]
        else:
            bar_colors = '#1d4ed8'

        if not obs.empty:
            fig1.add_trace(go.Bar(
                x=obs['date'], y=obs['precipitation'],
                name='Actual', marker_color=bar_colors, marker_line_width=0, opacity=0.85,
            ))
        if not fcst.empty:
            fig1.add_trace(go.Bar(
                x=fcst['date'], y=fcst['precipitation'],
                name='Forecast', marker_color='#ea580c', marker_line_width=0, opacity=0.75,
            ))

        fig1.update_layout(**PLOTLY_LAYOUT)
        fig1.update_layout(
            title=dict(text=f"{label} — Daily Precip  (mm/day)", font=dict(size=12)),
            yaxis_title="mm/day", height=270, barmode='overlay',
            margin=dict(l=55, r=10, t=45, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: cumulative departure from normal ──────────────────────────────
    with ch2:
        fig2 = go.Figure()
        obs_cum_end = 0.0

        if not obs.empty and 'mean_precip' in obs.columns:
            obs_s = obs.sort_values('date').copy()
            obs_s['dep'] = obs_s['precipitation'] - obs_s['mean_precip']
            obs_s['cum'] = obs_s['dep'].cumsum()
            obs_cum_end  = float(obs_s['cum'].iloc[-1])
            fig2.add_trace(go.Scatter(
                x=obs_s['date'], y=obs_s['cum'],
                mode='lines', name='Cumulative departure',
                line=dict(color='#1d4ed8', width=2.5),
                fill='tozeroy', fillcolor='rgba(29,78,216,0.10)',
            ))

        if not fcst.empty and 'mean_precip' in fcst.columns:
            fcst_s = fcst.sort_values('date').copy()
            fcst_s['dep'] = fcst_s['precipitation'] - fcst_s['mean_precip']
            fcst_s['cum'] = obs_cum_end + fcst_s['dep'].cumsum()
            bridge = (pd.DataFrame({'date': [obs_s['date'].iloc[-1]], 'cum': [obs_cum_end]})
                      if not obs.empty else pd.DataFrame())
            fcst_plot = pd.concat([bridge, fcst_s[['date', 'cum']]])
            fig2.add_trace(go.Scatter(
                x=fcst_plot['date'], y=fcst_plot['cum'],
                mode='lines', name='Forecast (extended)',
                line=dict(color='#ea580c', width=2, dash='dash'),
            ))

        fig2.add_hline(y=0, line_color='#94a3b8', line_width=1)
        fig2.update_layout(**PLOTLY_LAYOUT)
        fig2.update_layout(
            title=dict(text=f"{label} — Cumulative Departure from Normal", font=dict(size=12)),
            yaxis_title="Deviation (mm)", height=270,
            margin=dict(l=55, r=10, t=45, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_gatun_lake_chart(hist_df: pd.DataFrame, proj_df: pd.DataFrame):
    """Observed Gatun Lake levels + official ACP forecast + critical threshold lines.

    Mirrors notebook's gatun_lake_basic.py: blue observed, red forecast,
    orange dashed at 85 ft (increased rates), dark-red dashed at 82 ft (restrictions).
    """
    if hist_df.empty:
        st.warning("No Gatun Lake level data — ACP endpoint may be unreachable.")
        return

    today = pd.Timestamp.today()
    one_yr_ago = today - pd.DateOffset(days=365)
    recent = hist_df[hist_df['date'] >= one_yr_ago]

    # KPI: current level
    current_level = float(recent.sort_values('date')['level_ft'].iloc[-1]) if not recent.empty else None
    if current_level is not None:
        if current_level < 82:
            status, delta_color = "Critical — Major Restrictions", "inverse"
        elif current_level < 85:
            status, delta_color = "Caution — Rate Surcharges", "inverse"
        else:
            status, delta_color = "Normal Operations", "normal"
        kc1, kc2 = st.columns([1, 3])
        with kc1:
            st.metric("Gatun Lake Level", f"{current_level:.1f} ft", delta=status,
                      delta_color=delta_color)

    fig = go.Figure()

    if not recent.empty:
        fig.add_trace(go.Scatter(
            x=recent['date'], y=recent['level_ft'],
            mode='lines', name='Observed',
            line=dict(color='#1d4ed8', width=2),
        ))

    if not proj_df.empty:
        fig.add_trace(go.Scatter(
            x=proj_df['date'], y=proj_df['level_ft'],
            mode='lines', name='Official ACP Forecast',
            line=dict(color='#dc2626', width=2, dash='dash'),
        ))

    fig.add_hline(y=85, line_color='#f59e0b', line_dash='dash', line_width=1.5,
                  annotation_text='85 ft — Increased Shipping Rates',
                  annotation_font_color='#b45309', annotation_position='bottom right')
    fig.add_hline(y=82, line_color='#7f1d1d', line_dash='dash', line_width=1.5,
                  annotation_text='82 ft — Significant Restrictions',
                  annotation_font_color='#7f1d1d', annotation_position='bottom right')

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f'Gatun Lake — Observed & Official Forecast  (as of {today:%Y-%m-%d})',
            font=dict(size=13),
        ),
        yaxis_title='Lake Level (feet)',
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_map():
    st.caption("Gridded forecast deviation from ERA5 climatology (2000–2024). Data: ECMWF-ENS.")

    import matplotlib
    matplotlib.use('Agg')
    try:
        import cartopy.crs as ccrs          # noqa: F401
        import cartopy.feature as cfeature  # noqa: F401
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        map_region = st.selectbox("Region", list(MAP_REGIONS.keys()), key="map_region")
    with ctrl2:
        variable = st.radio("Variable", ["Temperature", "Precipitation"],
                            horizontal=True, key="map_var")

    bounds   = MAP_REGIONS[map_region]
    today_d  = datetime.now().date()
    periods  = [
        (today_d + timedelta(days=i * 5),
         today_d + timedelta(days=(i + 1) * 5 - 1))
        for i in range(3)
    ]

    overlay = _draw_three_gorges if (map_region == 'East Asia' and variable == 'Precipitation') else None

    # ── Three maps side by side ────────────────────────────────────────────────
    map_cols = st.columns(3)
    for i, (p_start, p_end) in enumerate(periods):
        period_label = f"{p_start:%d %b} – {p_end:%d %b}"
        with map_cols[i]:
            with st.spinner(f"Days {i*5+1}–{(i+1)*5}…"):
                if variable == "Temperature":
                    try:
                        df = load_gridded_anomalies_multiday(map_region, p_start, p_end)
                    except Exception as e:
                        st.error(str(e)); continue
                    if df.empty:
                        st.warning("No data"); continue
                    lats_s = np.sort(df['latitude'].unique())
                    lons_s = np.sort(df['longitude'].unique())
                    data_2d = (df.pivot_table(index='latitude', columns='longitude', values='anomaly')
                                 .reindex(index=lats_s, columns=lons_s).values)
                    fig_mpl = _make_map_fig(
                        lons_s, lats_s, data_2d, bounds, period_label,
                        'RdBu_r', np.linspace(-10, 10, 21), 'T2m Anomaly (°C)',
                        has_cartopy, figsize=(5, 4),
                    )
                else:
                    try:
                        df = load_gridded_precip_deviation(map_region, p_start, p_end)
                    except Exception as e:
                        st.error(str(e)); continue
                    if df.empty:
                        st.warning("No data"); continue
                    lats_s = np.sort(df['latitude'].unique())
                    lons_s = np.sort(df['longitude'].unique())
                    data_2d = (df.pivot_table(index='latitude', columns='longitude', values='anomaly')
                                 .reindex(index=lats_s, columns=lons_s).values)
                    fig_mpl = _make_map_fig(
                        lons_s, lats_s, data_2d, bounds, period_label,
                        'BrBG', np.linspace(-8, 8, 17), 'Precip Deviation (mm/day)',
                        has_cartopy, figsize=(5, 4), extra_draw=overlay,
                    )

            st.pyplot(fig_mpl, use_container_width=True)
            import matplotlib.pyplot as plt
            plt.close(fig_mpl)

    # ── Three Gorges / Yangtze catchment (East Asia + Precipitation) ──────────
    if map_region == 'East Asia' and variable == 'Precipitation':
        st.markdown("---")
        st.markdown("#### THREE GORGES CATCHMENT — YANGTZE RIVER")
        st.caption("Area-average precipitation · upstream Yangtze catchment (~1M km²) · "
                   "deviation from ERA5 2000–2024 climatology")
        try:
            tg_hist, tg_fcst, tg_clim = load_watershed_precip('Three Gorges')
            _render_watershed_charts('Three Gorges', tg_hist, tg_fcst, tg_clim)
        except Exception as e:
            st.error(f"Three Gorges data: {e}")




# ─── Tab 5: CDD Forecast ─────────────────────────────────────────────────────────
def render_cdd_forecast():
    st.markdown("#### CDD FORECAST")
    st.caption("14-day ECMWF-ENS forecast vs climatology (2000–2024). CDD = max(T_mean − 18°C, 0).")

    selected = st.multiselect(
        "Regions", list(REGION_MAP.keys()), DEFAULT_REGIONS,
        label_visibility="collapsed", key="fcst_regions",
    )
    if not selected:
        st.info("Select at least one region.")
        return

    summary_rows = []
    cols = st.columns(2)

    for idx, region in enumerate(selected):
        try:
            hist_df = load_historical(region)
            fcst_df = load_forecast(region)

            if fcst_df.empty:
                with cols[idx % 2]:
                    st.warning(f"No forecast data for {region}.")
                continue

            clim_temp = compute_temperature_climatology_simple(hist_df)
            clim_cdd = compute_daily_cdd_climatology_v2(hist_df)

            fcst_temp = fcst_df[['date', 'temperature']].copy()
            fcst_cdd = fcst_df[['date', 'cdd']].copy()

            fig_temp = make_forecast_temperature_chart(region, fcst_temp, clim_temp)
            fig_dev = make_forecast_cdd_deviation_chart(region, fcst_cdd, clim_cdd)

            with cols[idx % 2]:
                st.plotly_chart(fig_temp, use_container_width=True)
                st.plotly_chart(fig_dev, use_container_width=True)

            # Summary stats for KPI cards
            if not fcst_cdd.empty and not clim_cdd.empty:
                fc = fcst_cdd.copy()
                fc['day_of_year'] = pd.to_datetime(fc['date']).dt.day_of_year
                merged = fc.merge(clim_cdd, on='day_of_year', how='left')
                total_fcst = merged['cdd'].sum()
                total_normal = merged['mean_cdd'].fillna(0).sum()
                deviation = total_fcst - total_normal
                summary_rows.append({
                    'Region': region,
                    'Fcst CDD': f"{total_fcst:.0f}",
                    'Normal CDD': f"{total_normal:.0f}",
                    'Deviation': f"{deviation:+.0f}",
                    'Days': len(merged),
                })
        except Exception as e:
            with cols[idx % 2]:
                st.error(f"{region}: {e}")
                with st.expander("Details"):
                    st.text(traceback.format_exc())

    if summary_rows:
        st.markdown("---")
        st.markdown("#### FORECAST SUMMARY")
        kpi_cols = st.columns(min(len(summary_rows), 6))
        for i, row in enumerate(summary_rows[:6]):
            dev = float(row['Deviation'])
            cls = "kpi-card-warm" if dev > 0 else "kpi-card-cool"
            with kpi_cols[i]:
                st.markdown(kpi_card(row['Region'], dev, "°C·d vs normal", card_class=cls), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ─── Tab: Gatun Lake ──────────────────────────────────────────────────────────────
def render_gatun_lake():
    st.markdown("#### GATUN LAKE — PANAMA CANAL")
    st.caption("Observed water levels + official ACP forecast · "
               "source: Panama Canal Authority (evtms-rpts.pancanal.com)")
    try:
        gt_hist, gt_proj = load_gatun_lake_levels()
        _render_gatun_lake_chart(gt_hist, gt_proj)
    except Exception as e:
        st.error(f"Gatun data unavailable: {e}")


# ─── Tab: Hurricanes ─────────────────────────────────────────────────────────────

_HURRICANE_COLORS = {
    'TD':    '#9ca3af',
    'TS':    '#16a34a',
    'Cat 1': '#eab308',
    'Cat 2': '#f97316',
    'Cat 3': '#ef4444',
    'Cat 4': '#b91c1c',
    'Cat 5': '#7c3aed',
}

_BASIN_LABELS = {
    'Atlantic':     'ATL',
    'E.Pacific':    'EP',
    'W.Pacific':    'WP',
    'Indian Ocean': 'IO',
    'S.Pacific':    'SP',
}


def render_hurricanes():
    st.caption("Active tropical cyclones · NHC (Atlantic / E.Pacific) + JTWC via NOAA tgftp (W.Pacific / Indian Ocean / S.Pacific) · ECMWF AIFS if eccodes installed")

    with st.spinner("Fetching live hurricane data…"):
        try:
            result = load_hurricane_data()
            storms, sources = result if isinstance(result, tuple) else (result, {})
        except Exception as e:
            st.error(f"Hurricane data unavailable: {e}")
            return

    # ── Data source status ────────────────────────────────────────────────────
    src_parts = []
    has_errors = False
    for src, status in sources.items():
        if status in ('ok', 'skipped') or status.startswith('ok'):
            src_parts.append(f"**{src.upper()}** ✓")
        else:
            src_parts.append(f"**{src.upper()}** ✗")
            has_errors = True
    if src_parts:
        st.caption("Sources: " + "  ·  ".join(src_parts))

    if has_errors or not storms:
        with st.expander("Data source diagnostics", expanded=not storms):
            for src, status in sources.items():
                if status not in ('ok', 'skipped') and not status.startswith('ok'):
                    st.error(f"**{src.upper()}**: {status}")
                else:
                    st.success(f"**{src.upper()}**: {status}")
            st.write(f"Total storms loaded: **{len(storms)}**")

    # ── Map (always shown, even with no active storms) ────────────────────────
    fig = go.Figure()

    if storms:
        for storm in storms:
            color = _HURRICANE_COLORS.get(storm['category'], '#6b7280')

            # Forecast track — dashed line + small dots
            track = storm.get('forecast_track', [])
            if track:
                track_lons = [storm['lon']] + [p['lon'] for p in track]
                track_lats = [storm['lat']] + [p['lat'] for p in track]
                track_winds = [storm['wind_kt']] + [p['wind_kt'] for p in track]
                track_hrs   = ['Now'] + [f"+{p['hours']}h" for p in track]
                fig.add_trace(go.Scattergeo(
                    lon=track_lons, lat=track_lats,
                    mode='lines+markers',
                    line=dict(color=color, width=1.5, dash='dot'),
                    marker=dict(
                        size=[12] + [6] * len(track),
                        color=[_HURRICANE_COLORS.get(_kt_to_category_display(w), color)
                               for w in track_winds],
                        opacity=0.75,
                    ),
                    text=track_hrs,
                    hovertemplate='%{text}<br>%{lat:.1f}°N %{lon:.1f}°E<extra></extra>',
                    showlegend=False,
                ))

            # Current position — large marker with name label
            wind = storm['wind_kt']
            size = max(14, min(28, 10 + wind // 5))
            fig.add_trace(go.Scattergeo(
                lon=[storm['lon']], lat=[storm['lat']],
                mode='markers+text',
                marker=dict(
                    size=size, color=color,
                    symbol='circle',
                    line=dict(width=2, color='#0f172a'),
                ),
                text=[storm['name']],
                textposition='top center',
                textfont=dict(size=10, color='#0f172a', family='Inter, sans-serif'),
                name=f"{storm['name']}  {storm['category']}  {wind} kt  [{_BASIN_LABELS.get(storm['basin'], storm['basin'])}]",
                hovertemplate=(
                    f"<b>{storm['name']}</b>  ({storm['classification']})<br>"
                    f"Basin: {storm['basin']}<br>"
                    f"Category: {storm['category']}<br>"
                    f"Wind: {wind} kt<br>"
                    f"Pressure: {storm.get('pressure_mb', 'N/A')} mb<br>"
                    f"Last update: {storm.get('last_update', 'N/A')}<extra></extra>"
                ),
            ))

    fig.update_layout(
        geo=dict(
            projection_type='natural earth',
            showland=True,        landcolor='#e2e8f0',
            showocean=True,       oceancolor='#dbeafe',
            showcountries=True,   countrycolor='#94a3b8',
            showcoastlines=True,  coastlinecolor='#475569',
            showlakes=False,
            resolution=50,
            showframe=False,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family='Inter, sans-serif', color='#0f172a', size=11),
        legend=dict(
            bgcolor='rgba(255,255,255,0.92)',
            font=dict(size=10, color='#374151'),
            bordercolor='#e2e8f0',
            borderwidth=1,
            x=0.01, y=0.01, xanchor='left', yanchor='bottom',
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    if not storms:
        st.info("No active tropical cyclones reported at this time.")
        return

    # ── Category legend ───────────────────────────────────────────────────────
    cat_cols = st.columns(7)
    for i, (cat, col) in enumerate(_HURRICANE_COLORS.items()):
        with cat_cols[i]:
            st.markdown(
                f'<div style="background:{col};color:#fff;border-radius:6px;'
                f'padding:4px 8px;text-align:center;font-size:0.75rem;'
                f'font-weight:600;font-family:Inter,sans-serif">{cat}</div>',
                unsafe_allow_html=True,
            )

    # ── Details table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ACTIVE STORMS")
    rows = []
    for s in storms:
        rows.append({
            'Name':          s['name'],
            'Basin':         s['basin'],
            'Category':      s['category'],
            'Wind (kt)':     s['wind_kt'],
            'Pressure (mb)': s.get('pressure_mb', 'N/A'),
            'Lat':           f"{s['lat']:.1f}°{'N' if s['lat'] >= 0 else 'S'}",
            'Lon':           f"{abs(s['lon']):.1f}°{'E' if s['lon'] >= 0 else 'W'}",
            'Forecast pts':  len(s.get('forecast_track', [])),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _kt_to_category_display(wind_kt: int) -> str:
    """Re-export for use inside render_hurricanes (avoid circular issue)."""
    if wind_kt < 34:    return 'TD'
    elif wind_kt < 64:  return 'TS'
    elif wind_kt < 83:  return 'Cat 1'
    elif wind_kt < 96:  return 'Cat 2'
    elif wind_kt < 113: return 'Cat 3'
    elif wind_kt < 137: return 'Cat 4'
    else:               return 'Cat 5'


# ─── Tab: Coal Brief ─────────────────────────────────────────────────────────────

_BRIEF_CSS = """
<style>
.brief-box {
    background: #f8fafc;
    border: 1px solid rgba(15,23,42,0.10);
    border-left: 3px solid #1d4ed8;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 8px 0 16px;
}
.brief-bullet {
    padding: 6px 0;
    font-size: 0.91rem;
    color: #0f172a;
    line-height: 1.55;
    border-bottom: 1px solid rgba(15,23,42,0.06);
}
.brief-bullet:last-child { border-bottom: none; }
.brief-bullet b { color: #1d4ed8; }
</style>
"""


def _compute_cdd_summary(regions: list) -> dict:
    """Compute last-14-day CDD total vs 2000-2024 normal for the given regions.

    Uses load_current_year_cdd() for actual recent data (ERA5 + ECMWF forecast),
    and the precomputed historical table for the normal baseline.
    """
    today = pd.Timestamp.today().normalize()
    current_year = today.year
    season_start = pd.Timestamp(f"{current_year}-04-15")
    window_start = today - pd.Timedelta(days=14)
    summary = {}

    # Load all historical CDD once — filters per-region inside the loop
    try:
        precomp_hist = load_precomputed_historical()
    except Exception:
        precomp_hist = pd.DataFrame()

    for region in regions:
        try:
            # ── Recent CDD: ERA5 actuals + ECMWF forecast gap-fill ───────────────
            current_df = load_current_year_cdd(region)
            if current_df.empty:
                continue
            current_df = current_df.copy()
            current_df["date"] = pd.to_datetime(current_df["date"])
            recent = current_df[current_df["date"] >= window_start]
            if recent.empty:
                continue
            current_14d = float(recent["cdd"].sum())

            # ── Normal: cumulative difference from 2000-2024 climatology ─────────
            normal_14d = 0.0
            dos_end = max(1, (today - season_start).days)
            dos_start = max(0, dos_end - 14)

            if not precomp_hist.empty and "region" in precomp_hist.columns and dos_end > 0:
                region_hist = precomp_hist[precomp_hist["region"] == region][["date", "cdd"]].copy()
                if not region_hist.empty:
                    normal_df = compute_normal(region_hist)
                    if not normal_df.empty:
                        # compute_normal returns CUMULATIVE mean; take the 14-day increment
                        row_end   = normal_df.loc[normal_df["day_of_season"] == dos_end,   "mean"].values
                        row_start = normal_df.loc[normal_df["day_of_season"] == dos_start, "mean"].values
                        if len(row_end) and len(row_start):
                            normal_14d = float(row_end[0]) - float(row_start[0])
                        elif len(row_end):
                            normal_14d = float(row_end[0])

            summary[region] = {
                "current_7d": round(current_14d, 1),
                "anomaly":    round(current_14d - normal_14d, 1),
            }
        except Exception:
            continue

    return summary


def _get_az_credentials() -> tuple[str, str, str] | tuple[None, None, None]:
    """Try Databricks secrets → st.secrets → environment variables."""
    import os

    # 1. Databricks secrets (dbutils available in Databricks runtimes)
    try:
        t = dbutils.secrets.get("axpo", "azure_tenant_id")   # type: ignore[name-defined]
        c = dbutils.secrets.get("axpo", "azure_client_id")   # type: ignore[name-defined]
        s = dbutils.secrets.get("axpo", "azure_client_secret")  # type: ignore[name-defined]
        if t and c and s:
            return t, c, s
    except Exception:
        pass

    # 2. Databricks SDK (Databricks Apps context)
    try:
        from databricks.sdk.runtime import dbutils as sdk_dbutils
        t = sdk_dbutils.secrets.get("axpo", "azure_tenant_id")
        c = sdk_dbutils.secrets.get("axpo", "azure_client_id")
        s = sdk_dbutils.secrets.get("axpo", "azure_client_secret")
        if t and c and s:
            return t, c, s
    except Exception:
        pass

    # 3. Streamlit secrets
    try:
        t = st.secrets["azure_tenant_id"]
        c = st.secrets["azure_client_id"]
        s = st.secrets["azure_client_secret"]
        if t and c and s:
            return t, c, s
    except Exception:
        pass

    # 4. Environment variables
    t = os.environ.get("AZURE_TENANT_ID")
    c = os.environ.get("AZURE_CLIENT_ID")
    s = os.environ.get("AZURE_CLIENT_SECRET")
    if t and c and s:
        return t, c, s

    return None, None, None


def render_coal_brief():
    st.markdown("#### AI COAL MARKET BRIEF")
    st.caption(
        "6-agent AI analysis: storm supply risk · Rhine/Kaub transport · "
        "European CDD · Asia-Pacific CDD · China Three Gorges hydro → coal market outlook"
    )
    st.markdown(_BRIEF_CSS, unsafe_allow_html=True)

    tenant_id, client_id, client_secret = _get_az_credentials()
    if not all([tenant_id, client_id, client_secret]):
        st.warning(
            "Azure OpenAI credentials not found. "
            "Configure `azure_tenant_id`, `azure_client_id`, `azure_client_secret` "
            "in Databricks secrets (scope: **axpo**), `st.secrets`, or environment variables."
        )
        return

    col_txt, col_btn = st.columns([5, 1])
    with col_btn:
        generate = st.button("Generate", type="primary", key="coal_brief_btn")

    cached = st.session_state.get("coal_brief_result")

    if not generate and not cached:
        st.info(
            "Click **Generate** to run the 6-agent AI analysis. "
            "Each run makes 6 Azure OpenAI calls (~20-30 seconds)."
        )
        with st.expander("What each agent uses"):
            st.markdown("""
| Agent | Data source |
|---|---|
| Hurricane | NHC + JTWC live storm data (already loaded by this app) |
| Kaub | Live Rhine gauge from Pegelonline (German Federal Waterways) |
| European CDD | Germany, France — Databricks ERA5 + ECMWF forecast |
| Asia-Pacific CDD | China N/C/S, Japan, South Korea, India — Databricks ERA5 + ECMWF forecast |
| China hydro | Three Gorges catchment precipitation from Databricks ERA5/ECMWF |
| Synthesis | Combines all five findings into 5-6 coal trader bullets |
""")
        return

    if generate:
        from _ai_coal_brief import EU_REGIONS, ASIA_REGIONS
        progress = st.empty()

        def _prog(msg: str):
            progress.caption(f"⟳  {msg}")

        try:
            _prog("Loading storm data…")
            storms_result = load_hurricane_data()
            storms, _ = storms_result if isinstance(storms_result, tuple) else (storms_result, {})

            _prog("Computing European CDD anomalies…")
            cdd_eu = {}
            try:
                cdd_eu = _compute_cdd_summary(EU_REGIONS)
            except Exception:
                pass

            _prog("Computing Asia-Pacific CDD anomalies…")
            cdd_asia = {}
            try:
                cdd_asia = _compute_cdd_summary(ASIA_REGIONS)
            except Exception:
                pass

            _prog("Loading Three Gorges watershed precipitation…")
            tg_hist, tg_fcst, tg_clim = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            try:
                tg_hist, tg_fcst, tg_clim = load_watershed_precip("Three Gorges")
            except Exception:
                pass

            from _ai_coal_brief import generate_coal_brief as _run

            result = _run(
                storms=storms,
                three_gorges_hist=tg_hist,
                three_gorges_fcst=tg_fcst,
                three_gorges_clim=tg_clim,
                cdd_eu=cdd_eu,
                cdd_asia=cdd_asia,
                azure_tenant_id=tenant_id,
                azure_client_id=client_id,
                azure_client_secret=client_secret,
                progress_cb=_prog,
            )
            st.session_state["coal_brief_result"] = result
            cached = result

        except Exception as exc:
            progress.empty()
            st.error(f"Brief generation failed: {exc}")
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())
            return

        progress.empty()

    if not cached:
        return

    from _ai_coal_brief import brief_to_html_bullets

    kaub_str = (f" · Kaub {cached['kaub_level_cm']:.0f} cm"
                if cached.get("kaub_level_cm") else "")
    st.markdown(f"**Coal Market Brief**{kaub_str} · {cached['generated_at']}")

    st.markdown(
        f'<div class="brief-box">{brief_to_html_bullets(cached["synthesis"])}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### SPECIALIST AGENT OUTPUTS")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Hurricane / storm supply risk"):
            st.markdown(cached["hurricane"])
        with st.expander("European CDD / gas-coal switching"):
            st.markdown(cached.get("cdd_eu", ""))
        with st.expander("China Three Gorges hydro"):
            st.markdown(cached["china_hydro"])
    with c2:
        kaub_exp = (f"Rhine / Kaub — {cached['kaub_level_cm']:.0f} cm"
                    if cached.get("kaub_level_cm") else "Rhine / Kaub")
        with st.expander(kaub_exp):
            st.markdown(cached["kaub"])
        with st.expander("Asia-Pacific CDD / coal power demand"):
            st.markdown(cached.get("cdd_asia", ""))


# ─── Tab: Kaub Levels ────────────────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False)
def _fetch_kaub_pdf() -> bytes:
    import requests as _req
    r = _req.get(
        "https://vorhersage.bafg.de/14-Tage-Vorhersage/Kaub_14Tage.pdf",
        headers={"User-Agent": "Mozilla/5.0 (compatible; CDD-Dashboard/1.0)"},
        timeout=30,
    )
    r.raise_for_status()
    return r.content


def render_kaub_levels():
    st.markdown("#### KAUB — RHINE WATER LEVEL FORECAST")
    st.caption(
        "14-day water level forecast at the Kaub gauge (Rhine km 546) · "
        "Source: BfG — German Federal Institute of Hydrology"
    )
    with st.spinner("Loading Kaub forecast…"):
        try:
            pdf_bytes = _fetch_kaub_pdf()
        except Exception as e:
            st.error(f"Could not load Kaub forecast: {e}")
            st.markdown(
                "[Open Kaub 14-day forecast PDF directly ↗](https://vorhersage.bafg.de/14-Tage-Vorhersage/Kaub_14Tage.pdf)"
            )
            return

    import base64 as _b64
    b64 = _b64.b64encode(pdf_bytes).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="880" style="border:none;border-radius:8px;" '
        f'type="application/pdf"></iframe>',
        unsafe_allow_html=True,
    )
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name="Kaub_14Tage.pdf",
        mime="application/pdf",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("Coal Desk CDD")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Maps & Watersheds",
        "CDD Dashboard",
        "Gatun Lake",
        "Kaub Levels",
        "CDD Forecast",
        "Hurricanes",
        "Coal Brief",
    ])
    with tab1:
        render_anomaly_map()
    with tab2:
        render_cdd_dashboard()
    with tab3:
        render_gatun_lake()
    with tab4:
        render_kaub_levels()
    with tab5:
        render_cdd_forecast()
    with tab6:
        render_hurricanes()
    with tab7:
        render_coal_brief()


if __name__ == "__main__":
    main()
