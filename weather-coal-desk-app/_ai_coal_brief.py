"""
Coal market AI brief — 6-agent architecture.

Flow
----
  Python data collectors build compact, structured context documents.
  Five independent specialist LLM calls each analyse one domain.
  One synthesis LLM call reads all five findings and writes 4-6 trader bullets.

Agents
------
  1. hurricane    — storm threats to coal supply chains
  2. kaub         — Rhine water level and European coal barge transport
  3. cdd_eu       — European CDD anomalies and gas-coal switching
  4. cdd_asia     — Asia-Pacific CDD and direct coal power demand
  5. china_hydro  — Three Gorges catchment hydro and China coal demand
  6. synthesis    — combines all five findings into a coal market brief
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd


# ── Region lists (match _config.REGION_MAP keys) ─────────────────────────────

EU_REGIONS   = ['Germany', 'France']
ASIA_REGIONS = ['China North', 'China Central', 'China South', 'Japan', 'South Korea', 'India']


# ── Kaub navigation thresholds (cm above chart datum) ────────────────────────

KAUB_THRESHOLDS = {
    "crisis":      78,   # below: Rhine coal transport effectively halted
    "severe":     130,   # below: 40-60% capacity reduction
    "constrained": 200,  # below: ~20% draft restriction
}


# ── System prompts ─────────────────────────────────────────────────────────────

_SYS_HURRICANE = """\
You are a coal market analyst specialising in supply-chain disruption from tropical storms.

Key coal export terminals by basin:
  W.Pacific / Indian Ocean:
    Newcastle NSW (Australia) — world's largest coal export port
    Dalrymple Bay / Hay Point QLD (Australia)
    Kalimantan / Balikpapan / Samarinda (Indonesia)
  Atlantic / Americas:
    Puerto Drummond & Cerrejón (Colombia) — API#2 ARA supply
    Hampton Roads (USA)
  Indian Ocean / Africa:
    Richards Bay (South Africa) — API#4 benchmark

Major dry-bulk shipping lanes:
  W.Pacific transit routes (Australia → NE Asia, Indonesia → China/India)
  Indian Ocean (Richards Bay → India/Europe)
  Gulf of Mexico / Caribbean (US, Colombia → NWE, Mediterranean)

Given the list of active tropical cyclones:
1. Identify any storm threatening a coal terminal or major shipping lane.
2. State the directional impact on coal benchmarks (Newcastle, API#4, API#2 ARA).
3. If no relevant coal-market threat exists, say so in ONE concise line.

Format: each bullet starts with "- ". Name the storm, intensity, location,
and the specific terminal or route at risk.

Last line only — write exactly one of:
SIGNAL: BULLISH  (storm threatens a coal terminal or key shipping lane — net bullish for prices)
SIGNAL: BEARISH  (no relevant coal-market threat from current storms)
SIGNAL: NEUTRAL  (minor or uncertain coal-market impact)
"""

_SYS_KAUB = """\
You are a European energy analyst specialising in Rhine inland waterway transport.

Kaub gauge (Rhine km 546) is the critical chokepoint for coal, fuel oil, and
chemical barge traffic between ARA ports and German industrial consumers.

Navigation thresholds:
  ≥ 200 cm  normal draft, full capacity, standard freight rates
  130–200   draft-restricted, ~20% capacity loss, freight +15-25%
  78–130    severely restricted, 40-60% capacity loss, significant ARA-inland spread
  < 78 cm   near-total disruption (2022 crisis level)

Market impact:
  Low Rhine → barges can't move coal inland → ARA stock builds → depresses ARA spot
              OR → inland shortage → Rhine freight premium → supports delivered prices
  Rising Rhine after restriction → relief rally in inland freight, ARA re-draws

Write 1-2 bullet points covering:
  - Current level, directional trend, and 14-day outlook
  - Bullish or bearish for API#2 ARA coal or European coal-vs-gas spreading

Format: each bullet starts with "- ". Quote the specific level and the threshold context.

Last line only — write exactly one of:
SIGNAL: BULLISH  (Rhine constraining coal barge transport — supports ARA coal prices)
SIGNAL: BEARISH  (Rhine at full capacity — no transport constraint premium)
SIGNAL: NEUTRAL  (situation mixed or not significantly restrictive)
"""

_SYS_CDD_EU = """\
You are a European power and gas analyst. CDD = max(T_mean − 18°C, 0).
High CDD = above-normal heat = more air-conditioning = more electricity demand.
In summer: high CDD → more gas-for-power → TTF bid up → coal-gas switching point rises.
In winter: cold anomaly → more heating demand → more gas burn → bullish TTF and coal.

Regions: Germany, France. Context: Benelux, UK, Iberia, Italy (not always in data).

Write 1-2 bullet points on:
  - Dominant temperature / CDD signal this week and next 1-2 weeks
  - Whether it drives TTF / gas demand higher or lower
  - Coal switching implication: is coal gaining or losing to gas?

Format: each bullet starts with "- ". Quote specific countries and CDD magnitudes.

Last line only — write exactly one of:
SIGNAL: BULLISH  (heat raising European gas demand or supporting coal-gas switching toward coal)
SIGNAL: BEARISH  (cool / below-normal temperatures reducing energy demand)
SIGNAL: NEUTRAL  (temperature signal mixed or marginal for coal)
"""

_SYS_CDD_ASIA = """\
You are an Asia-Pacific power and coal demand analyst.
CDD = max(T_mean − 18°C, 0). In Asia, cooling demand is met primarily by coal-fired power.
China, Japan, and South Korea together import ~400 Mt/year of thermal coal.

High CDD in China (esp. North, Central) → air-conditioning load up → coal power up
  → bearish thermal coal stocks → bullish Newcastle, Indonesian HBA imports.
High CDD in Japan/South Korea → similar dynamic (LNG + coal-fired power).
High CDD in India → more coal-fired power → bullish Indian imports (Indonesian HBA).

Write 1-2 bullet points on:
  - The dominant temperature / CDD signal across China, Japan, South Korea, India
  - Direct coal demand implication (power burn), NOT gas switching (Asia burns coal directly)
  - Bullish or bearish for Newcastle index and Indonesian HBA

Format: each bullet starts with "- ". Separate China from Japan/Korea/India where signals differ.

Last line only — write exactly one of:
SIGNAL: BULLISH  (above-normal heat driving coal power demand — bullish Newcastle / HBA)
SIGNAL: BEARISH  (below-normal temperatures reducing coal power burn)
SIGNAL: NEUTRAL  (temperature signal mild or offsetting across regions)
"""

_SYS_CHINA_HYDRO = """\
You are a China energy analyst specialising in power generation and coal demand.

Three Gorges Dam / Yangtze basin context:
  Installed capacity:  22.5 GW (largest single hydro plant globally)
  Normal annual output: ~90-100 TWh
  Reservoir range:     145 m (flood drawdown) to 175 m (full supply level)
  Filling season:      Jun–Sep (Yangtze monsoon / South China Sea moisture)
  Drawdown season:     Nov–Apr (power demand, irrigation)

Rule of thumb:
  Each 10 GW shortfall from Three Gorges ≈ +25 Mt/year equivalent coal demand.
  China coal imports (Australian, Indonesian): ~300 Mt/year.

Given Yangtze basin precipitation anomaly data (from ERA5/ECMWF for the catchment):
  Positive precip anomaly → above-normal inflows → more hydro → bearish China coal
  Negative precip anomaly → deficient inflows → less hydro → bullish China coal

Write 1-2 bullet points on:
  - Current Three Gorges catchment hydro situation (precip anomaly + seasonal context)
  - Directional impact on China coal demand and Asia-Pacific benchmarks
    (Newcastle index, Indonesian HBA)

Format: each bullet starts with "- ".

Last line only — write exactly one of:
SIGNAL: BULLISH  (deficient inflows / reduced hydro → China needs more coal)
SIGNAL: BEARISH  (above-normal inflows / excess hydro → China coal demand reduced)
SIGNAL: NEUTRAL  (near-normal catchment hydro, marginal coal-demand impact)
"""

_SYS_SYNTHESIS = """\
You are the head coal trader at a major European energy company writing the daily brief.
You have received analyses from five specialist agents:
  1. Hurricane/storm risk to coal supply chains
  2. Rhine/Kaub water levels and European inland coal transport
  3. European CDD anomalies and gas-coal switching
  4. Asia-Pacific CDD and direct coal power demand
  5. China Three Gorges hydro and China coal demand

Write exactly 5-6 bullet points for the morning coal market brief.

Requirements:
  - LEAD with the single most market-moving signal right now.
  - Cover BOTH the European supply/transport angle (API#2 ARA) and the
    Asian demand angle (Newcastle, Indonesian HBA) — at least one bullet each.
  - One bullet must describe a cross-market interaction where two or more signals
    reinforce or offset each other (e.g., "Asia CDD surge + low Rhine = bullish API#2 + Newcastle").
  - One stability bullet: primary uncertainty or what would flip this view.
  - No intro sentence. No headers. No conclusion. No numbering.
  - Bullets only, each starting with "- ".

Last line only — write: SIGNAL: BULLISH, SIGNAL: BEARISH, or SIGNAL: NEUTRAL
(net coal market direction synthesising all five agent signals).
"""


# ── Data formatters (pure Python, no LLM) ─────────────────────────────────────

def _fmt_hurricanes(storms: list) -> str:
    ts = datetime.utcnow().strftime("%d %b %Y %H:%M UTC")
    if not storms:
        return f"No active tropical cyclones in any monitored basin. ({ts})"

    lines = [f"Active tropical cyclones — {ts}", f"Total storms: {len(storms)}", ""]
    for s in storms:
        name = s.get("name", "Unknown")
        basin = s.get("basin", "Unknown")
        cat = s.get("category", "TS")
        wind = s.get("wind_kt", 0)
        lat = s.get("lat", 0.0)
        lon = s.get("lon", 0.0)
        pres = s.get("pressure_hpa")

        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        line = (f"  {name} ({basin}) — {cat}, {wind} kt, "
                f"{abs(lat):.1f}°{ns} {abs(lon):.1f}°{ew}")
        if pres:
            line += f", {pres} hPa"
        lines.append(line)

        track = s.get("forecast_track", [])
        for pt in track[:4]:
            h = pt.get("hours", 0)
            if h == 0:
                continue
            plat, plon = pt.get("lat", 0.0), pt.get("lon", 0.0)
            pw = pt.get("wind_kt", 0)
            pns = "N" if plat >= 0 else "S"
            pew = "E" if plon >= 0 else "W"
            lines.append(f"    +{h:02d}h: {abs(plat):.1f}°{pns} {abs(plon):.1f}°{pew} {pw} kt")

    return "\n".join(lines)


def _fmt_kaub(measurements: list, current_cm: float | None) -> str:
    ts = datetime.utcnow().strftime("%d %b %Y")
    lines = [f"Kaub Rhine gauge — {ts}"]

    if current_cm is not None:
        lines.append(f"Current level: {current_cm:.0f} cm")
        if current_cm < KAUB_THRESHOLDS["crisis"]:
            lines.append("Status: CRITICAL — below 78 cm, Rhine coal transport halted")
        elif current_cm < KAUB_THRESHOLDS["severe"]:
            lines.append("Status: SEVERELY RESTRICTED (78–130 cm) — 40-60% capacity reduction")
        elif current_cm < KAUB_THRESHOLDS["constrained"]:
            lines.append("Status: CONSTRAINED (130–200 cm) — ~20% capacity reduction")
        else:
            lines.append("Status: NORMAL (≥200 cm) — full barge operations")
    else:
        lines.append("Current level: data unavailable")

    if measurements:
        vals = [m["value"] for m in measurements if m.get("value") is not None]
        if len(vals) >= 2:
            trend = vals[-1] - vals[0]
            t_str = (f"rising +{trend:.0f} cm" if trend > 5 else
                     f"falling {trend:.0f} cm" if trend < -5 else "stable")
            lines.append(f"Trend over data period: {t_str}")
            lines.append(f"Period range: {min(vals):.0f}–{max(vals):.0f} cm")

    lines.append("14-day forecast: BfG vorhersage.bafg.de/14-Tage-Vorhersage/Kaub_14Tage.pdf")
    return "\n".join(lines)


def _fmt_cdd(cdd_summary: dict, label: str = "") -> str:
    """cdd_summary: {region: {"anomaly": float, "current_7d": float}}"""
    ts = datetime.utcnow().strftime("%d %b %Y")
    header = f"{'(' + label + ') ' if label else ''}CDD anomalies — last 14 days vs 2000-2024 normal — {ts}"

    if not cdd_summary:
        month = datetime.utcnow().month
        ctx = ("CDD season active (Jun–Sep)" if month in (6, 7, 8, 9) else
               "Shoulder season — CDD near zero" if month in (4, 5, 10, 11) else
               "Winter — heating season, CDD minimal")
        return f"{header}\nNo detailed CDD data available. Seasonal context: {ctx}"

    lines = [header, f"{'Region':<22} {'14d CDD':>10} {'Anomaly':>10}  Direction"]
    for region, d in sorted(cdd_summary.items()):
        anom = d.get("anomaly", 0.0)
        curr = d.get("current_7d", 0.0)
        direction = "warmer than normal" if anom > 0.5 else ("colder than normal" if anom < -0.5 else "near normal")
        lines.append(f"  {region:<22} {curr:>10.1f} {anom:>+10.1f}  {direction}")
    return "\n".join(lines)


def _fmt_china_hydro(
    hist_df: pd.DataFrame,
    fcst_df: pd.DataFrame,
    clim_df: pd.DataFrame,
) -> str:
    ts = datetime.utcnow().strftime("%d %b %Y")
    month = datetime.utcnow().month

    season_map = {
        (6, 7, 8, 9):  "Peak monsoon/filling season — Three Gorges inflows typically HIGH",
        (10, 11):      "Post-monsoon drawdown — reservoir releasing water, still HIGH",
        (12, 1, 2, 3): "Dry season — reservoir at seasonal minimum, hydro generation LOW",
        (4, 5):        "Pre-monsoon — low levels, inflows beginning to rise",
    }
    season_ctx = next((v for ks, v in season_map.items() if month in ks), "")

    lines = [
        f"Three Gorges Catchment Hydro — {ts}",
        f"Seasonal context: {season_ctx}",
        "",
        "Area-averaged precipitation for the Three Gorges catchment basin:",
    ]

    if not hist_df.empty and not clim_df.empty:
        hist_df = hist_df.copy()
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        today = pd.Timestamp.today().normalize()
        recent = hist_df[hist_df["date"] >= today - pd.Timedelta(days=30)]

        if not recent.empty:
            total_30d = recent["precipitation"].sum()
            lines.append(f"  Observed 30-day total (catchment avg): {total_30d:.1f} mm")

            # Compare to climatology
            clim_df = clim_df.copy()
            doys = recent["date"].dt.dayofyear.values
            clim_sub = clim_df[clim_df["doy"].isin(doys)]
            if not clim_sub.empty:
                normal_30d = clim_sub["mean_precip"].sum()
                if normal_30d > 0:
                    anom_pct = (total_30d - normal_30d) / normal_30d * 100
                    direction = "above" if anom_pct > 0 else "below"
                    lines.append(f"  vs ERA5 climatology: {anom_pct:+.0f}% ({direction} normal, "
                                 f"normal = {normal_30d:.1f} mm)")
                    if anom_pct > 20:
                        lines.append("  Assessment: Elevated catchment inflows — bullish hydro output, bearish China coal")
                    elif anom_pct < -20:
                        lines.append("  Assessment: Deficient inflows — reduced hydro output, bullish China coal demand")
                    else:
                        lines.append("  Assessment: Near-normal inflows — hydro output broadly seasonal")

    if not fcst_df.empty and not clim_df.empty:
        fcst_df = fcst_df.copy()
        fcst_df["date"] = pd.to_datetime(fcst_df["date"])
        fcst_14d = fcst_df.head(14)
        if not fcst_14d.empty:
            fcst_total = fcst_14d["precipitation"].sum()
            lines.append(f"  ECMWF 14-day forecast (catchment avg): {fcst_total:.1f} mm")
            doys_fcst = fcst_14d["date"].dt.dayofyear.values
            clim_fcst = clim_df[clim_df["doy"].isin(doys_fcst)]
            if not clim_fcst.empty:
                normal_fcst = clim_fcst["mean_precip"].sum()
                if normal_fcst > 0:
                    anom_fcst = (fcst_total - normal_fcst) / normal_fcst * 100
                    lines.append(f"  Forecast 14-day anomaly: {anom_fcst:+.0f}% vs normal")

    lines.extend([
        "",
        "Reference:",
        "  Three Gorges capacity: 22.5 GW",
        "  China coal imports: ~300 Mt/year (Australian + Indonesian)",
        "  Rule of thumb: 10 GW hydro shortfall ≈ +25 Mt/year coal demand equivalent",
    ])
    return "\n".join(lines)


# ── Signal extractor ──────────────────────────────────────────────────────────

def _extract_signal(text: str) -> tuple[str, str]:
    """Remove SIGNAL: line(s) from agent output and return (signal, cleaned_text)."""
    signal = "NEUTRAL"
    clean = []
    for line in text.strip().split("\n"):
        m = re.match(r"^SIGNAL:\s*(BULLISH|BEARISH|NEUTRAL)\s*$", line.strip(), re.IGNORECASE)
        if m:
            signal = m.group(1).upper()
        else:
            clean.append(line)
    # strip trailing blank lines
    while clean and not clean[-1].strip():
        clean.pop()
    return signal, "\n".join(clean)


# ── Kaub data fetcher ──────────────────────────────────────────────────────────

def fetch_kaub_levels() -> tuple[list, float | None]:
    """Fetch Kaub gauge from Pegelonline (German federal waterways REST API)."""
    import requests
    base = "https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations/KAUB/W"

    try:
        r = requests.get(f"{base}/measurements.json", params={"start": "P30D"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data:
            return data, float(data[-1]["value"])
    except Exception:
        pass

    try:
        r = requests.get(f"{base}/currentmeasurement.json", timeout=15)
        r.raise_for_status()
        d = r.json()
        v = float(d["value"])
        return [{"timestamp": d.get("timestamp"), "value": v}], v
    except Exception:
        return [], None


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_client(tenant_id: str, client_id: str, client_secret: str):
    from azure.identity import ClientSecretCredential
    from openai import AzureOpenAI

    cred = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )
    return AzureOpenAI(
        azure_endpoint="https://azure-oai-prod.openai.azure.com/",
        api_version="2024-12-01-preview",
        azure_ad_token_provider=lambda: cred.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token,
    )


def _llm(client, system: str, user_doc: str, model: str, max_tokens: int = 400) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_doc},
        ],
        temperature=0.3,
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def generate_coal_brief(
    storms: list,
    three_gorges_hist: pd.DataFrame,
    three_gorges_fcst: pd.DataFrame,
    three_gorges_clim: pd.DataFrame,
    cdd_eu: dict,
    cdd_asia: dict,
    azure_tenant_id: str,
    azure_client_id: str,
    azure_client_secret: str,
    model: str = "gpt-4o",
    progress_cb=None,
) -> dict:
    """Run the 6-agent coal brief pipeline.

    Parameters
    ----------
    storms               list of storm dicts from load_hurricane_data()
    three_gorges_hist    hist_df from load_watershed_precip("Three Gorges")
    three_gorges_fcst    fcst_df from load_watershed_precip("Three Gorges")
    three_gorges_clim    clim_df from load_watershed_precip("Three Gorges")
    cdd_eu               {region: {"anomaly": float, "current_7d": float}} for European regions
    cdd_asia             {region: {"anomaly": float, "current_7d": float}} for Asian regions
    azure_*              service-principal credentials for Azure OpenAI
    model                Azure OpenAI deployment name (default "gpt-4o")
    progress_cb          optional callable(str) for Streamlit progress messages

    Returns
    -------
    dict — hurricane, kaub, cdd_eu, cdd_asia, china_hydro, synthesis,
           kaub_level_cm, generated_at
    """
    def _prog(msg: str):
        if progress_cb:
            progress_cb(msg)

    client = _build_client(azure_tenant_id, azure_client_id, azure_client_secret)

    # ── Step 1: fetch live data ───────────────────────────────────────────────
    _prog("Fetching Kaub Rhine gauge data…")
    kaub_measurements, kaub_level = fetch_kaub_levels()

    # ── Step 2: format context documents ─────────────────────────────────────
    doc_hurricane  = _fmt_hurricanes(storms)
    doc_kaub       = _fmt_kaub(kaub_measurements, kaub_level)
    doc_cdd_eu     = _fmt_cdd(cdd_eu,   "Europe")
    doc_cdd_asia   = _fmt_cdd(cdd_asia, "Asia-Pacific")
    doc_hydro      = _fmt_china_hydro(three_gorges_hist, three_gorges_fcst, three_gorges_clim)

    # ── Step 3: specialist agents (independent, sequential) ───────────────────
    _prog("Hurricane analyst…")
    sig_hurricane, brief_hurricane = _extract_signal(
        _llm(client, _SYS_HURRICANE, doc_hurricane, model))

    _prog("Kaub analyst…")
    sig_kaub, brief_kaub = _extract_signal(
        _llm(client, _SYS_KAUB, doc_kaub, model))

    _prog("European CDD analyst…")
    sig_cdd_eu, brief_cdd_eu = _extract_signal(
        _llm(client, _SYS_CDD_EU, doc_cdd_eu, model))

    _prog("Asia-Pacific CDD analyst…")
    sig_cdd_asia, brief_cdd_asia = _extract_signal(
        _llm(client, _SYS_CDD_ASIA, doc_cdd_asia, model))

    _prog("China hydro analyst…")
    sig_hydro, brief_hydro = _extract_signal(
        _llm(client, _SYS_CHINA_HYDRO, doc_hydro, model))

    # ── Step 4: synthesis ─────────────────────────────────────────────────────
    _prog("Synthesis agent — writing coal market brief…")
    synthesis_input = "\n\n".join([
        "=== STORM / HURRICANE SUPPLY RISK ===",        brief_hurricane,
        "=== RHINE / KAUB TRANSPORT LEVELS ===",        brief_kaub,
        "=== EUROPEAN CDD / GAS-COAL SWITCHING ===",    brief_cdd_eu,
        "=== ASIA-PACIFIC CDD / COAL POWER DEMAND ===", brief_cdd_asia,
        "=== CHINA THREE GORGES HYDRO ===",             brief_hydro,
    ])
    sig_synthesis, brief_synthesis = _extract_signal(
        _llm(client, _SYS_SYNTHESIS, synthesis_input, model, max_tokens=700))

    return {
        "hurricane":     brief_hurricane,
        "kaub":          brief_kaub,
        "cdd_eu":        brief_cdd_eu,
        "cdd_asia":      brief_cdd_asia,
        "china_hydro":   brief_hydro,
        "synthesis":     brief_synthesis,
        "signals": {
            "hurricane":   sig_hurricane,
            "kaub":        sig_kaub,
            "cdd_eu":      sig_cdd_eu,
            "cdd_asia":    sig_cdd_asia,
            "china_hydro": sig_hydro,
            "synthesis":   sig_synthesis,
        },
        "kaub_level_cm": kaub_level,
        "generated_at":  datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
    }


# ── Display helper ─────────────────────────────────────────────────────────────

def brief_to_html_bullets(brief_text: str) -> str:
    """Convert LLM markdown bullets to styled HTML divs."""
    parts = []
    for line in brief_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        parts.append(f'<div class="brief-bullet">• {line}</div>')
    return "\n".join(parts)
