"""Plotly chart builders — Coal Desk CDD Dashboard."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from _style import PLOTLY_LAYOUT
from _config import BASE_TEMP

# Colors for the 5 most-similar historical years — saturated for white background
_SIMILAR_COLORS = ['#1d4ed8', '#7c3aed', '#ea580c', '#0369a1', '#16a34a']

# Per-model colours and legend labels for the CDD Forecast tab
_MODEL_COLORS = {'ecmwf-ens': '#ea580c', 'ecmwf-vareps': '#7c3aed'}
_MODEL_LABELS = {'ecmwf-ens': 'ENS (14d)', 'ecmwf-vareps': 'vareps (44d)'}


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared dark-navy Plotly theme."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ─── Cumulative CDD chart ────────────────────────────────────────────────────────

def make_cumulative_cdd_chart(
    region: str,
    cum_current: pd.DataFrame,
    cum_prev: pd.DataFrame,
    normal: pd.DataFrame,
    current_year: int,
    all_historical_cumulative: dict = None,
    similar_years: list = None,
    ensemble_spread: pd.DataFrame = None,
) -> go.Figure:
    """Build the cumulative CDD chart for one region.

    Mirrors the notebook style: faint grey background for all historical years,
    5 most-similar years highlighted in color, current year in bold red.

    Args:
        all_historical_cumulative: dict {year: cum_df} pre-computed for 2000-2024.
        similar_years: list of (year, score) tuples from compute_similar_years(),
                       sorted best-first.
    """
    fig = go.Figure()
    prev_year = current_year - 1
    similar_year_set = {y for y, _ in (similar_years or [])}

    # ── All historical years as faint grey background lines ──────────────────
    if all_historical_cumulative:
        for year, cum_df in sorted(all_historical_cumulative.items()):
            if year == current_year or year in similar_year_set or cum_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=cum_df['day_of_season'], y=cum_df['cumulative_cdd'],
                mode='lines',
                line=dict(color='rgba(107,114,128,0.18)', width=1),
                showlegend=False, hoverinfo='skip',
            ))

    # ── 5 most-similar historical years ──────────────────────────────────────
    if similar_years and all_historical_cumulative:
        for rank, (year, score) in enumerate(similar_years):
            cum_df = all_historical_cumulative.get(year)
            if cum_df is None or cum_df.empty:
                continue
            color = _SIMILAR_COLORS[rank % len(_SIMILAR_COLORS)]
            fig.add_trace(go.Scatter(
                x=cum_df['day_of_season'], y=cum_df['cumulative_cdd'],
                mode='lines',
                line=dict(color=color, width=1.8, dash='dot'),
                name=f'{year}  (#{rank + 1} similar)',
                opacity=0.85,
            ))

    # ── Normal band (mean ± 1σ) ───────────────────────────────────────────────
    if not normal.empty:
        fig.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['upper'],
            mode='lines', line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['lower'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(107,114,128,0.12)',
            name='Normal ±1σ  (2000–2024)',
        ))
        fig.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['mean'],
            mode='lines', line=dict(color='#9ca3af', dash='dash', width=1.5),
            name='Normal mean',
        ))

    # ── Previous year ─────────────────────────────────────────────────────────
    if not cum_prev.empty and prev_year not in similar_year_set:
        fig.add_trace(go.Scatter(
            x=cum_prev['day_of_season'], y=cum_prev['cumulative_cdd'],
            mode='lines', line=dict(color='#6b7280', width=1.5, dash='dash'),
            name=str(prev_year),
        ))

    # ── Current year  (actual solid + forecast dashed) ───────────────────────
    if not cum_current.empty:
        today = pd.Timestamp.today().normalize()
        actual = cum_current[cum_current['date'] <= today]
        forecast = cum_current[cum_current['date'] > today]

        if not actual.empty:
            fig.add_trace(go.Scatter(
                x=actual['day_of_season'], y=actual['cumulative_cdd'],
                mode='lines', line=dict(color='#dc2626', width=3),
                name=f'{current_year} (Actual)',
            ))

        if not forecast.empty:
            connect = pd.concat([actual.tail(1), forecast]) if not actual.empty else forecast
            fig.add_trace(go.Scatter(
                x=connect['day_of_season'], y=connect['cumulative_cdd'],
                mode='lines', line=dict(color='#dc2626', width=2, dash='dash'),
                name=f'{current_year} (Forecast)',
            ))

    # ── Ensemble spread band (shaded uncertainty around forecast) ─────────────
    if ensemble_spread is not None and not ensemble_spread.empty:
        fig.add_trace(go.Scatter(
            x=ensemble_spread['day_of_season'], y=ensemble_spread['cumulative_upper'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=ensemble_spread['day_of_season'], y=ensemble_spread['cumulative_lower'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(220,38,38,0.12)',
            name='ENS 25–75th pct',
        ))

    fig.update_layout(
        title=dict(text=f"Cumulative CDD — {region}", font=dict(size=14, color='#0f172a')),
        xaxis=dict(title="Days since April 15", range=[0, 150]),
        yaxis_title="Cumulative CDD (°C·d)",
        height=400,
        legend=dict(
            x=1.01, y=1, xanchor='left',
            font=dict(size=10, color='#374151'),
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#e2e8f0',
            borderwidth=1,
        ),
    )
    return _apply_theme(fig)


# ─── Temperature time series ────────────────────────────────────────────────────

def make_temperature_chart(city: str, city_data: pd.DataFrame) -> go.Figure:
    """Daily temperature time series with forecast extension."""
    fig = go.Figure()
    today = pd.Timestamp.today().normalize()
    hist = city_data[city_data['date'] <= today]
    fcst = city_data[city_data['date'] > today]

    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist['date'], y=hist['temperature'],
            mode='lines', name='Observed',
            line=dict(color='#1d4ed8', width=1.5),
        ))
    if not fcst.empty:
        connect = pd.concat([hist.tail(1), fcst]) if not hist.empty else fcst
        fig.add_trace(go.Scatter(
            x=connect['date'], y=connect['temperature'],
            mode='lines', name='Forecast (ENS)',
            line=dict(color='#ea580c', width=2, dash='dash'),
        ))

    fig.add_hline(y=BASE_TEMP, line_dash='dot', line_color='#16a34a',
                  annotation_text=f"CDD base ({BASE_TEMP}°C)",
                  annotation_font_color='#16a34a')

    fig.update_layout(
        title=dict(text=f"Daily Mean Temperature — {city}", font=dict(size=14)),
        xaxis_title="Date", yaxis_title="°C",
        height=340,
    )
    return _apply_theme(fig)


# ─── Daily CDD bar chart ─────────────────────────────────────────────────────────

def make_daily_cdd_bars(city: str, city_data: pd.DataFrame) -> go.Figure:
    """Bar chart of daily CDD (last 30 days + forecast)."""
    fig = go.Figure()
    today = pd.Timestamp.today().normalize()
    df = city_data.copy()
    df['cdd'] = (df['temperature'] - BASE_TEMP).clip(lower=0)
    recent = df[df['date'] >= today - timedelta(days=30)]

    hist_r = recent[recent['date'] <= today]
    fcst_r = recent[recent['date'] > today]

    if not hist_r.empty:
        fig.add_trace(go.Bar(
            x=hist_r['date'], y=hist_r['cdd'],
            name='Actual', marker_color='#1d4ed8',
            marker_line_width=0,
        ))
    if not fcst_r.empty:
        fig.add_trace(go.Bar(
            x=fcst_r['date'], y=fcst_r['cdd'],
            name='Forecast', marker_color='#ea580c',
            marker_line_width=0,
        ))

    fig.update_layout(
        title=dict(text=f"Daily CDD — {city}", font=dict(size=14)),
        xaxis_title="Date", yaxis_title="CDD (°C·d)",
        height=280, barmode='stack',
    )
    return _apply_theme(fig)


# ─── CDD Forecast charts ─────────────────────────────────────────────────────────

def make_forecast_temperature_chart(region: str, fcst_temp: pd.DataFrame, clim_temp: pd.DataFrame) -> go.Figure:
    """Forecast temperature vs climatological normal. Handles multi-model data (model column optional)."""
    fig = go.Figure()

    if not fcst_temp.empty:
        # Build climatology band over the full date range across all models
        all_dates = fcst_temp[['date']].drop_duplicates().sort_values('date').copy()
        all_dates['day_of_year'] = pd.to_datetime(all_dates['date']).dt.day_of_year

        if not clim_temp.empty:
            cband = all_dates.merge(clim_temp, on='day_of_year', how='left').sort_values('date')
            fig.add_trace(go.Scatter(
                x=cband['date'], y=cband['mean_temp'] + cband['std_temp'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=cband['date'], y=(cband['mean_temp'] - cband['std_temp']).clip(lower=0),
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(107,114,128,0.15)',
                name='Clim ±1σ',
            ))
            fig.add_trace(go.Scatter(
                x=cband['date'], y=cband['mean_temp'],
                mode='lines', line=dict(color='#9ca3af', dash='dash', width=1.5),
                name='Clim mean',
            ))

        if 'model' in fcst_temp.columns:
            for model in ['ecmwf-ens', 'ecmwf-vareps']:
                mdf = fcst_temp[fcst_temp['model'] == model].sort_values('date')
                if mdf.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=mdf['date'], y=mdf['temperature'],
                    mode='lines', name=_MODEL_LABELS.get(model, model),
                    line=dict(color=_MODEL_COLORS.get(model, '#ea580c'), width=2.5,
                              dash='solid' if model == 'ecmwf-ens' else 'dash'),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=fcst_temp['date'], y=fcst_temp['temperature'],
                mode='lines', name='Forecast',
                line=dict(color='#ea580c', width=2.5),
            ))

    fig.add_hline(y=BASE_TEMP, line_dash='dot', line_color='#16a34a',
                  annotation_text=f"CDD base ({BASE_TEMP}°C)",
                  annotation_font_color='#16a34a')

    fig.update_layout(
        title=dict(text=f"Temperature Forecast — {region}", font=dict(size=13, color='#0f172a')),
        xaxis_title=None, yaxis_title="°C",
        height=310,
        margin=dict(l=50, r=120, t=40, b=35),
        legend=dict(x=1.01, y=1, xanchor='left', font=dict(size=10, color='#374151'),
                    bgcolor='rgba(255,255,255,0.92)', bordercolor='#e2e8f0', borderwidth=1),
    )
    return _apply_theme(fig)


def make_forecast_cdd_deviation_chart(region: str, fcst_cdd: pd.DataFrame, clim_cdd: pd.DataFrame) -> go.Figure:
    """Daily CDD deviation from normal (forecast − climatology). Handles multi-model data."""
    fig = go.Figure()

    if not fcst_cdd.empty and not clim_cdd.empty:
        if 'model' in fcst_cdd.columns:
            for model in ['ecmwf-ens', 'ecmwf-vareps']:
                mdf = fcst_cdd[fcst_cdd['model'] == model].copy().sort_values('date')
                if mdf.empty:
                    continue
                mdf['day_of_year'] = pd.to_datetime(mdf['date']).dt.day_of_year
                merged = mdf.merge(clim_cdd, on='day_of_year', how='left')
                merged['mean_cdd'] = merged['mean_cdd'].fillna(0)
                merged['deviation'] = merged['cdd'] - merged['mean_cdd']
                warm_col = _MODEL_COLORS.get(model, '#ea580c')
                cool_col = '#2563eb' if model == 'ecmwf-ens' else '#6d28d9'
                bar_colors = [warm_col if d >= 0 else cool_col for d in merged['deviation']]
                fig.add_trace(go.Bar(
                    x=merged['date'], y=merged['deviation'],
                    name=_MODEL_LABELS.get(model, model),
                    marker_color=bar_colors, marker_line_width=0,
                    opacity=0.85 if model == 'ecmwf-ens' else 0.45,
                ))
        else:
            df = fcst_cdd.copy()
            df['day_of_year'] = pd.to_datetime(df['date']).dt.day_of_year
            merged = df.merge(clim_cdd, on='day_of_year', how='left')
            merged['mean_cdd'] = merged['mean_cdd'].fillna(0)
            merged['deviation'] = merged['cdd'] - merged['mean_cdd']
            bar_colors = ['#dc2626' if d >= 0 else '#2563eb' for d in merged['deviation']]
            fig.add_trace(go.Bar(
                x=merged['date'], y=merged['deviation'],
                name='Deviation', marker_color=bar_colors, marker_line_width=0, opacity=0.85,
            ))
    elif not fcst_cdd.empty:
        src = fcst_cdd[fcst_cdd['model'] == 'ecmwf-ens'] if 'model' in fcst_cdd.columns else fcst_cdd
        fig.add_trace(go.Bar(x=src['date'], y=src['cdd'], name='CDD', marker_color='#ea580c'))

    fig.add_hline(y=0, line_color='#94a3b8', line_width=1)

    fig.update_layout(
        title=dict(text=f"CDD Anomaly vs Normal — {region}", font=dict(size=13, color='#0f172a')),
        xaxis_title=None, yaxis_title="°C·d deviation",
        height=250, barmode='overlay',
        margin=dict(l=50, r=120, t=40, b=35),
        legend=dict(x=1.01, y=1, xanchor='left', font=dict(size=10, color='#374151'),
                    bgcolor='rgba(255,255,255,0.92)', bordercolor='#e2e8f0', borderwidth=1),
    )
    return _apply_theme(fig)


# ─── Anomaly map ─────────────────────────────────────────────────────────────────

def make_anomaly_map(anomalies: pd.DataFrame, target_date) -> go.Figure:
    """Scatter-geo map color-coded by temperature anomaly."""
    fig = px.scatter_geo(
        anomalies,
        lat='latitude', lon='longitude',
        color='anomaly',
        hover_name='city',
        hover_data={'temperature': ':.1f', 'climatology': ':.1f', 'anomaly': ':.1f'},
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        range_color=[-8, 8],
        size_max=12,
        title=f"Temperature Anomaly (°C) — {target_date}",
        projection='natural earth',
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.6, color='#ffffff')))
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#0f172a'),
        geo=dict(
            bgcolor='#dbeafe',
            landcolor='#e2e8f0',
            showland=True,
            showcountries=True,
            countrycolor='#94a3b8',
            coastlinecolor='#475569',
            showocean=True,
            oceancolor='#dbeafe',
            showlakes=False,
        ),
    )
    return fig
