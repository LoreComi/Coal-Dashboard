"""Shared visual styling — Coal Desk CDD Dashboard.

Design direction: clean professional light theme, slate-blue accents,
high-contrast text, suitable for Bloomberg-style trading screens.
"""

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Design tokens ─────────────────────────────────────────────────── */
    :root {
        --bg-app:         #f0f4f8;
        --bg-surface:     #ffffff;
        --bg-card:        #f8fafc;
        --bg-card-hover:  #f1f5f9;
        --border:         rgba(15,23,42,0.10);
        --border-bright:  rgba(15,23,42,0.22);
        --text-primary:   #0f172a;
        --text-secondary: #374151;
        --text-muted:     #6b7280;
        --axpo-red:       #dc2626;
        --accent:         #1d4ed8;
        --bull:           #dc2626;
        --bear:           #0369a1;
        --temp-warm:      #ea580c;
        --temp-cold:      #0284c7;
        --cdd-hot:        #ef4444;
        --cdd-cool:       #0ea5e9;
        --radius-sm:  6px;
        --radius-md:  10px;
        --radius-lg:  14px;
        --radius-pill: 999px;
        --shadow-card: 0 1px 6px rgba(15,23,42,0.07), 0 0 0 1px rgba(15,23,42,0.05);
        --shadow-pop:  0 6px 24px rgba(15,23,42,0.13);
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
        color: var(--text-muted) !important;
        font-weight: 700 !important;
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    p, .stMarkdown { color: var(--text-primary); }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 0.80rem !important;
    }
    a { color: var(--accent) !important; }

    /* ── Main content area ─────────────────────────────────────────── */
    .main .block-container {
        background: var(--bg-surface);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-card);
        padding: 24px 28px 32px !important;
        margin-top: 16px !important;
    }

    /* ── Tabs ───────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,23,42,0.05);
        border-radius: var(--radius-lg);
        padding: 4px;
        border-bottom: none !important;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-md) !important;
        font-weight: 500;
        padding: 8px 20px;
        font-size: 0.85rem;
        color: var(--text-secondary) !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.15s ease;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(15,23,42,0.07) !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--bg-surface) !important;
        color: var(--accent) !important;
        box-shadow: var(--shadow-card) !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
    .stTabs [data-baseweb="tab-border"]    { display: none !important; }

    /* ── KPI cards ──────────────────────────────────────────────── */
    .kpi-card {
        background: var(--bg-surface);
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
        background: linear-gradient(145deg, #fff7ed 0%, var(--bg-surface) 60%);
        border-color: rgba(234,88,12,0.22);
        border-top: 2px solid var(--temp-warm);
    }
    .kpi-card-cool {
        background: linear-gradient(145deg, #eff6ff 0%, var(--bg-surface) 60%);
        border-color: rgba(2,132,199,0.22);
        border-top: 2px solid var(--temp-cold);
    }
    .kpi-card-cdd {
        background: linear-gradient(145deg, #fef2f2 0%, var(--bg-surface) 60%);
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
        font-size: 0.62rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
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
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Inputs / selects ──────────────────────────────────────── */
    .stMultiSelect [data-baseweb="select"] > div,
    .stSelectbox  [data-baseweb="select"] > div {
        background: var(--bg-card) !important;
        border-color: var(--border) !important;
        border-radius: var(--radius-md) !important;
    }
    .stSlider [data-baseweb="slider"] {
        padding-top: 8px;
    }

    /* ── Data table ────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius-md) !important;
        overflow: hidden;
        border: 1px solid var(--border) !important;
    }

    /* ── Plotly charts ─────────────────────────────────────────── */
    .stPlotlyChart {
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-card);
        border: 1px solid var(--border);
        background: var(--bg-surface);
    }

    /* ── Pyplot / matplotlib images ────────────────────────────── */
    [data-testid="stImage"] > img {
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-card);
        border: 1px solid var(--border);
    }

    /* ── Dividers ──────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; }

    /* ── Section headings (markdown ####) ──────────────────────── */
    .stMarkdown h4 {
        margin-top: 24px !important;
        margin-bottom: 4px !important;
        color: var(--text-muted) !important;
    }
</style>
"""

# Plotly light theme — white plot area, slate text, subtle grid
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#ffffff',
    font=dict(family='Inter, sans-serif', color='#0f172a', size=12),
    xaxis=dict(
        gridcolor='#e2e8f0',
        zerolinecolor='#cbd5e1',
        linecolor='#e2e8f0',
        tickfont=dict(color='#6b7280', size=11),
        title_font=dict(color='#374151'),
    ),
    yaxis=dict(
        gridcolor='#e2e8f0',
        zerolinecolor='#cbd5e1',
        linecolor='#e2e8f0',
        tickfont=dict(color='#6b7280', size=11),
        title_font=dict(color='#374151'),
    ),
    margin=dict(l=55, r=20, t=55, b=45),
    legend=dict(
        bgcolor='rgba(255,255,255,0.92)',
        font=dict(size=11, color='#374151'),
        bordercolor='#e2e8f0',
        borderwidth=1,
    ),
)
