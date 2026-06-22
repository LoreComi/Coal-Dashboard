"""Shared visual styling — Coal Desk CDD Dashboard.

Design direction: dark-navy weather-intelligence platform,
matching the Morning Met Trader Dashboard aesthetic.
"""

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Design tokens ─────────────────────────────────────────────────── */
    :root {
        --bg-app:         #07111f;
        --bg-surface:     #0c1828;
        --bg-card:        #111f35;
        --bg-card-hover:  #172845;
        --border:         rgba(255,255,255,0.08);
        --border-bright:  rgba(255,255,255,0.16);
        --text-primary:   #e4eeff;
        --text-secondary: #7d9ab8;
        --text-muted:     #3d5470;
        --axpo-red:       #e2231a;
        --accent:         #2563eb;
        --bull:           #f87171;
        --bear:           #22d3ee;
        --temp-warm:      #fb923c;
        --temp-cold:      #60a5fa;
        --cdd-hot:        #ef4444;
        --cdd-cool:       #06b6d4;
        --radius-sm:  6px;
        --radius-md:  10px;
        --radius-lg:  14px;
        --radius-pill: 999px;
        --shadow-card: 0 2px 16px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
        --shadow-pop:  0 8px 32px rgba(0,0,0,0.55);
    }

    /* ── Base ──────────────────────────────────────────────────────── */
    .stApp {
        background: var(--bg-app) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    .stApp > header { background: transparent !important; }

    h1 {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        font-size: 1.8rem !important;
        margin-bottom: 0 !important;
        line-height: 1.2 !important;
    }
    h1::after {
        content: "";
        display: inline-block;
        width: 7px; height: 7px;
        margin-left: 8px;
        background: var(--axpo-red);
        border-radius: 2px;
        vertical-align: 5px;
    }
    h2 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        letter-spacing: -0.02em !important;
    }
    h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.0rem !important;
    }
    h4 {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    p, .stMarkdown, span, div { color: var(--text-primary); }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 0.80rem !important;
    }
    a { color: var(--accent) !important; }

    /* ── Tabs — pill container style ────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: var(--radius-lg);
        padding: 4px;
        border-bottom: none !important;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-md) !important;
        font-weight: 500;
        padding: 8px 18px;
        font-size: 0.84rem;
        color: var(--text-secondary) !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.15s ease;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255,255,255,0.06) !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-card) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
    .stTabs [data-baseweb="tab-border"]    { display: none !important; }

    /* ── KPI cards ──────────────────────────────────────────────── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 18px 20px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: var(--shadow-card);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        border-color: var(--border-bright);
        box-shadow: var(--shadow-pop);
    }
    .kpi-card-warm {
        background: linear-gradient(145deg, #1c1810 0%, var(--bg-card) 55%);
        border-color: rgba(251,146,60,0.22);
        border-top: 2px solid var(--temp-warm);
    }
    .kpi-card-cool {
        background: linear-gradient(145deg, #0e1a2e 0%, var(--bg-card) 55%);
        border-color: rgba(96,165,250,0.22);
        border-top: 2px solid var(--temp-cold);
    }
    .kpi-card-cdd {
        background: linear-gradient(145deg, #1f0e0e 0%, var(--bg-card) 55%);
        border-color: rgba(239,68,68,0.22);
        border-top: 2px solid var(--cdd-hot);
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 6px 0 4px;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.63rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-secondary);
    }
    .kpi-delta {
        font-size: 0.72rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .kpi-delta-up   { color: var(--bull); }
    .kpi-delta-down { color: var(--bear); }
    .kpi-delta-flat { color: var(--text-muted); }

    /* ── Sidebar ───────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: var(--text-secondary) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Data table ────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius-md) !important;
        overflow: hidden;
    }

    /* ── Plotly charts background ─────────────────────────────── */
    .stPlotlyChart {
        border-radius: var(--radius-md);
        overflow: hidden;
    }
</style>
"""

# Plotly dark theme matching the navy background
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#0c1828',
    font=dict(family='Inter, sans-serif', color='#e4eeff', size=12),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
)
