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
    load_watershed_precip, load_gatun_lake_levels, load_hurricane_data,
    THREE_GORGES_DAM,
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
    st.caption("Active tropical cyclones · NHC (Atlantic / E.Pacific) + JTWC (W.Pacific / Indian Ocean / S.Pacific)")

    with st.spinner("Fetching live hurricane data…"):
        try:
            result = load_hurricane_data()
            storms, sources = result if isinstance(result, tuple) else (result, {})
        except Exception as e:
            st.error(f"Hurricane data unavailable: {e}")
            return

    # ── Data source status ────────────────────────────────────────────────────
    src_parts = []
    for src, status in sources.items():
        if status == 'ok':
            src_parts.append(f"**{src.upper()}** ✓")
        elif status != 'skipped':
            src_parts.append(f"**{src.upper()}** ✗ `{status}`")
    if src_parts:
        st.caption("Sources: " + "  ·  ".join(src_parts))

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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("Coal Desk CDD")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Maps & Watersheds",
        "CDD Dashboard",
        "Gatun Lake",
        "City Detail",
        "Hurricanes",
    ])
    with tab1:
        render_anomaly_map()
    with tab2:
        render_cdd_dashboard()
    with tab3:
        render_gatun_lake()
    with tab4:
        render_city_detail()
    with tab5:
        render_hurricanes()


if __name__ == "__main__":
    main()
