"""Plotly chart builders — Coal Desk CDD Dashboard."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from _style import PLOTLY_LAYOUT
from _config import BASE_TEMP


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
) -> go.Figure:
    """Build the cumulative CDD fan chart for one region."""
    fig = go.Figure()
    prev_year = current_year - 1

    # Normal band (mean +/- 1 std)
    if not normal.empty:
        fig.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['upper'],
            mode='lines', line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['lower'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(100,116,139,0.18)',
            name='Normal ±1σ (2000–2024)',
        ))
        fig.add_trace(go.Scatter(
            x=normal['day_of_season'], y=normal['mean'],
            mode='lines', line=dict(color='#64748b', dash='dot', width=1.5),
            name='Normal Mean',
        ))

    # Previous year
    if not cum_prev.empty:
        fig.add_trace(go.Scatter(
            x=cum_prev['day_of_season'], y=cum_prev['cumulative_cdd'],
            mode='lines', line=dict(color='#60a5fa', width=1.5, dash='dash'),
            name=str(prev_year),
        ))

    # Current year (split actual / forecast)
    if not cum_current.empty:
        today = pd.Timestamp.today().normalize()
        actual = cum_current[cum_current['date'] <= today]
        forecast = cum_current[cum_current['date'] > today]

        if not actual.empty:
            fig.add_trace(go.Scatter(
                x=actual['day_of_season'], y=actual['cumulative_cdd'],
                mode='lines', line=dict(color='#ef4444', width=3),
                name=f'{current_year} (Actual)',
            ))

        if not forecast.empty:
            connect = pd.concat([actual.tail(1), forecast]) if not actual.empty else forecast
            fig.add_trace(go.Scatter(
                x=connect['day_of_season'], y=connect['cumulative_cdd'],
                mode='lines', line=dict(color='#ef4444', width=2, dash='dash'),
                name=f'{current_year} (Forecast)',
            ))

    fig.update_layout(
        title=dict(text=f"Cumulative CDD — {region}", font=dict(size=14)),
        xaxis_title="Days since April 15",
        yaxis_title="Cumulative CDD (°C·d)",
        height=380,
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
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
            line=dict(color='#60a5fa', width=1.5),
        ))
    if not fcst.empty:
        connect = pd.concat([hist.tail(1), fcst]) if not hist.empty else fcst
        fig.add_trace(go.Scatter(
            x=connect['date'], y=connect['temperature'],
            mode='lines', name='Forecast (ENS)',
            line=dict(color='#f87171', width=2, dash='dash'),
        ))

    fig.add_hline(y=BASE_TEMP, line_dash='dot', line_color='#22d3ee',
                  annotation_text=f"CDD base ({BASE_TEMP}°C)",
                  annotation_font_color='#22d3ee')

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
            name='Actual', marker_color='#60a5fa',
        ))
    if not fcst_r.empty:
        fig.add_trace(go.Bar(
            x=fcst_r['date'], y=fcst_r['cdd'],
            name='Forecast', marker_color='#fb923c',
        ))

    fig.update_layout(
        title=dict(text=f"Daily CDD — {city}", font=dict(size=14)),
        xaxis_title="Date", yaxis_title="CDD (°C·d)",
        height=280, barmode='stack',
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
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='rgba(255,255,255,0.3)')))
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0c1828',
        font=dict(family='Inter, sans-serif', color='#e4eeff'),
        geo=dict(
            bgcolor='#0c1828',
            landcolor='#1a2942',
            showland=True,
            showcountries=True,
            countrycolor='rgba(255,255,255,0.15)',
            coastlinecolor='rgba(255,255,255,0.2)',
            showocean=True,
            oceancolor='#07111f',
            showlakes=False,
        ),
    )
    return fig
