# ══════════════════════════════════════════════════════════════════════════════
#  Plataforma de Analítica — Comercio Exterior México
#  Streamlit app  |  3 módulos: Exportaciones · Importaciones · Clusters
# ══════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import statsmodels.api as sm

import os, io, base64

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeView MX",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#FFFFFF",
    "surface":  "#F5F7FA",
    "border":   "#DDE1E7",
    "text":     "#1A1F2E",
    "muted":    "#6B7280",
    "blue":     "#1A56DB",
    "green":    "#057A55",
    "orange":   "#C27803",
    "red":      "#E02424",
    "purple":   "#7E3AF2",
    "cyan":     "#0694A2",
}

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {C['bg']};
    color: {C['text']};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {C['surface']};
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{
    color: {C['text']} !important;
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 1rem 1.25rem;
}}
[data-testid="stMetricLabel"] {{ color: {C['muted']} !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; }}
[data-testid="stMetricValue"] {{ color: {C['text']} !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; }}
[data-testid="stMetricDelta"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }}

/* Headers */
h1 {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 1.6rem !important; color: {C['blue']} !important; letter-spacing: -.02em; }}
h2 {{ font-size: 1.15rem !important; color: {C['text']} !important; border-bottom: 1px solid {C['border']}; padding-bottom: .4rem; }}
h3 {{ font-size: .95rem !important; color: {C['muted']} !important; text-transform: uppercase; letter-spacing: .08em; }}

/* Buttons */
.stButton>button {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    color: {C['text']};
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .82rem;
    transition: all .15s;
}}
.stButton>button:hover {{
    border-color: {C['blue']};
    color: {C['blue']};
}}

/* Sliders / selects */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stRadio"] label {{
    color: {C['muted']} !important;
    font-size: .78rem !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: .82rem;
    color: {C['muted']};
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {C['blue']};
    border-bottom: 2px solid {C['blue']};
}}

/* Dataframe */
[data-testid="stDataFrame"] {{ border: 1px solid {C['border']}; border-radius: 8px; }}

/* Divider */
hr {{ border-color: {C['border']} !important; }}

/* Expander */
details summary {{ color: {C['muted']} !important; font-size: .82rem; }}

/* Remove Streamlit branding + hide sidebar completely */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="collapsedControl"] {{ display: none !important; }}
.block-container {{ padding-top: .5rem !important; }}

/* Plot bg override */
.js-plotly-plot .plotly {{ background: transparent !important; }}

/* Info / warning boxes */
.stAlert {{ background: {C['surface']}; border-color: {C['border']}; }}

/* Fix metric cards — force light background */
[data-testid="stMetric"] {{
    background: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
}}
[data-testid="metric-container"] {{
    background: {C['surface']} !important;
}}

/* Sidebar collapse button — always visible */
[data-testid="collapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: {C['blue']} !important;
    border-radius: 0 8px 8px 0 !important;
    color: white !important;
    width: 28px !important;
    top: 50% !important;
    box-shadow: 2px 2px 8px rgba(0,0,0,.15) !important;
}}
[data-testid="collapsedControl"]:hover {{
    background: #1346b8 !important;
}}

/* Main content never fully hides sidebar toggle */
section[data-testid="stSidebarContent"] {{
    background-color: {C['surface']};
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS (cached)
# ══════════════════════════════════════════════════════════════════════════════

BASE = os.path.dirname(__file__)

@st.cache_data
def load_trade_series(flow: str):
    fname = "EXPORTACIONES.xlsx" if flow == "Exportaciones" else "IMPORTACIONES.xlsx"
    df = pd.read_excel(os.path.join(BASE, fname))
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
    df = df.set_index("Month").sort_index()
    return df

@st.cache_data
def load_panel():
    df = pd.read_csv(os.path.join(BASE, "PANEL.csv"))
    df.columns = [c.strip() for c in df.columns]
    df["Trade_Value"] = pd.to_numeric(df["Trade_Value"], errors="coerce")
    df = df.dropna(subset=["Trade_Value"])
    df["Flow"] = df["Flow"].str.strip().str.title()
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFBFC",
    font=dict(family="IBM Plex Sans", color=C["text"], size=12),
    margin=dict(l=50, r=20, t=45, b=40),
    xaxis=dict(gridcolor="#E5E9F0", linecolor=C["border"], tickcolor=C["muted"]),
    yaxis=dict(gridcolor="#E5E9F0", linecolor=C["border"], tickcolor=C["muted"]),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["border"], borderwidth=1),
    hovermode="x unified",
)

def base_fig(**kw):
    fig = go.Figure()
    fig.update_layout(**LAYOUT_BASE, **kw)
    return fig

def fmt_usd(val):
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"

# ══════════════════════════════════════════════════════════════════════════════
#  TOP NAV — no sidebar, always visible
# ══════════════════════════════════════════════════════════════════════════════

PAGES = ["Resumen General", "Exportaciones", "Importaciones", "Clusters Municipios"]
ICONS = {"Resumen General": "📊", "Exportaciones": "📤", "Importaciones": "📥", "Clusters Municipios": "🗺️"}

if "page" not in st.session_state:
    st.session_state.page = "Resumen General"

# Top bar HTML
st.markdown(f"""
<div style="
    position:sticky;top:0;z-index:999;
    background:{C['bg']};
    border-bottom:2px solid {C['border']};
    padding:.6rem 2rem;
    display:flex;align-items:center;gap:2rem;
    margin-bottom:1.5rem;
">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:700;color:{C['blue']};white-space:nowrap;margin-right:.5rem">
    TradeView <span style="font-weight:300;color:{C['muted']};font-size:.75rem">MX</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Navigation buttons in one row
nav_cols = st.columns(len(PAGES))
for col, p in zip(nav_cols, PAGES):
    active = st.session_state.page == p
    with col:
        btn_style = f"""
        <style>
        div[data-testid="stButton"] button[kind="secondary"]#btn_{p.replace(' ','_')} {{
            border-bottom: 3px solid {C['blue']} !important;
        }}
        </style>
        """ if active else ""
        if st.button(
            f"{ICONS[p]}  {p}",
            key=f"nav_{p}",
            width="stretch",
            type="primary" if active else "secondary",
        ):
            st.session_state.page = p
            st.rerun()

st.markdown("<hr style='margin:.5rem 0 1.5rem 0'>", unsafe_allow_html=True)
page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE: RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════════════════

if page == "Resumen General":
    st.title("Resumen General")
    st.markdown(f"<p style='color:{C['muted']};margin-top:-.5rem'>Panorama del comercio exterior mexicano · Panel 2006–2025</p>", unsafe_allow_html=True)

    panel = load_panel()
    ts_exp = load_trade_series("Exportaciones")
    ts_imp = load_trade_series("Importaciones")

    # KPIs
    total_exp = ts_exp["Trade_Value"].sum()
    total_imp = ts_imp["Trade_Value"].sum()
    balance   = total_exp - total_imp
    municipios = panel["Municipio_ID"].nunique()

    # Crecimiento año anterior vs año más reciente (anual)
    def yoy_growth(ts):
        annual = ts["Trade_Value"].resample("YE").sum()
        if len(annual) >= 2:
            last, prev = annual.iloc[-1], annual.iloc[-2]
            return (last - prev) / prev * 100 if prev != 0 else None
        return None

    # Crecimiento total periodo completo
    yoy_exp   = yoy_growth(ts_exp)
    yoy_imp   = yoy_growth(ts_imp)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Exportaciones", fmt_usd(total_exp))
    c2.metric("Total Importaciones", fmt_usd(total_imp))
    c3.metric("Balance Comercial", fmt_usd(balance),
              delta="Superávit" if balance >= 0 else "Déficit",
              delta_color="normal" if balance >= 0 else "inverse")
    c4.metric("Municipios", f"{municipios}")

    # Fila de crecimiento
    st.markdown("")
    st.markdown("### 📈 Crecimiento del comercio exterior")
    g1, g2, g3, g4 = st.columns(4)
    if yoy_exp is not None:
        g1.metric("Exportaciones (YoY)", f"{yoy_exp:+.1f}%",
                  delta=f"{'▲' if yoy_exp >= 0 else '▼'} vs año anterior",
                  delta_color="normal" if yoy_exp >= 0 else "inverse")
    if yoy_imp is not None:
        g2.metric("Importaciones (YoY)", f"{yoy_imp:+.1f}%",
                  delta=f"{'▲' if yoy_imp >= 0 else '▼'} vs año anterior",
                  delta_color="normal" if yoy_imp >= 0 else "inverse")


    st.markdown("")

    # Balanza comercial mensual
    common = ts_exp.index.intersection(ts_imp.index)
    exp_c = ts_exp.loc[common, "Trade_Value"]
    imp_c = ts_imp.loc[common, "Trade_Value"]
    bal   = exp_c - imp_c

    tab1, tab2 = st.tabs(["📈  Series históricas", "📊  Balanza comercial"])

    with tab1:
        fig = base_fig(title="Exportaciones vs Importaciones (USD mensual)")
        fig.add_trace(go.Scatter(x=exp_c.index, y=exp_c.values, name="Exportaciones",
                                  line=dict(color=C["blue"], width=2)))
        fig.add_trace(go.Scatter(x=imp_c.index, y=imp_c.values, name="Importaciones",
                                  line=dict(color=C["orange"], width=2)))
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        fig2 = base_fig(title="Balanza Comercial Mensual (Exportaciones − Importaciones)")
        colors = [C["green"] if v >= 0 else C["red"] for v in bal.values]
        fig2.add_trace(go.Bar(x=bal.index, y=bal.values, marker_color=colors,
                               name="Balanza", opacity=0.85))
        fig2.add_hline(y=0, line_color=C["muted"], line_width=1)
        fig2.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig2, width="stretch")

    # Top estados
    st.markdown("### Top 10 estados por volumen total de comercio")
    top_states = (panel.groupby("State")["Trade_Value"]
                  .sum().sort_values(ascending=False).head(10).reset_index())
    fig3 = go.Figure(go.Bar(
        x=top_states["Trade_Value"] / 1e9,
        y=top_states["State"],
        orientation="h",
        marker=dict(color=C["blue"], opacity=0.85),
    ))
    _l3 = {**LAYOUT_BASE}
    _l3["yaxis"] = dict(autorange="reversed", **LAYOUT_BASE["yaxis"])
    fig3.update_layout(**_l3,
                       title="Trade Value total por Estado (Billones USD)",
                       xaxis_title="USD (Miles de millones)")
    st.plotly_chart(fig3, width="stretch")

    st.markdown("---")
    st.markdown(f"""
    <div style='padding:.75rem 1rem;background:{C['surface']};border:1px solid {C['border']};border-radius:8px;margin-top:.5rem'>
      <span style='font-size:.75rem;color:{C['muted']};font-family:IBM Plex Mono,monospace;font-weight:600;text-transform:uppercase;letter-spacing:.06em'>📌 Fuente de datos</span><br>
      <span style='font-size:.82rem;color:{C['text']}'>Secretaría de Economía — DataMéxico</span><br>
      <a href='https://www.economia.gob.mx/datamexico/es/profile/product/boneless-meat-of-bovine-animals-fresh-or-chilled'
         target='_blank'
         style='font-size:.78rem;color:{C['blue']};word-break:break-all'>
        economia.gob.mx/datamexico — Carne bovina deshuesada fresca o refrigerada
      </a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  REUSABLE TIME-SERIES MODULE
# ══════════════════════════════════════════════════════════════════════════════

def render_ts_module(flow_name: str, default_order, default_seasonal):
    color_main = C["blue"] if flow_name == "Exportaciones" else C["orange"]

    ts = load_trade_series(flow_name)
    y  = ts["Trade_Value"].astype(float).copy()
    if y.isna().any():
        y = y.interpolate(method="time")

    st.title(f"{'📤' if flow_name=='Exportaciones' else '📥'}  {flow_name}")
    st.markdown(f"<p style='color:{C['muted']};margin-top:-.5rem'>Serie mensual · {y.index.min().strftime('%b %Y')} – {y.index.max().strftime('%b %Y')} · {len(y)} observaciones</p>", unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor total acumulado", fmt_usd(y.sum()))
    c2.metric("Promedio mensual", fmt_usd(y.mean()))
    c3.metric("Máximo mensual", fmt_usd(y.max()))
    c4.metric("Mín. mensual", fmt_usd(y.min()))

    # KPIs de crecimiento
    annual_y = y.resample("YE").sum()
    if len(annual_y) >= 2:
        yoy_pct   = (annual_y.iloc[-1] - annual_y.iloc[-2]) / annual_y.iloc[-2] * 100
        last_yr   = annual_y.index[-1].year
        prev_yr   = annual_y.index[-2].year

        st.markdown("")
        g1, = st.columns(1)
        g1.metric(
            f"Crecimiento YoY ({prev_yr}→{last_yr})",
            f"{yoy_pct:+.1f}%",
            delta=f"{'Alza' if yoy_pct >= 0 else 'Baja'} vs año anterior",
            delta_color="normal" if yoy_pct >= 0 else "inverse"
        )

    # ── TABS ─────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📈  Serie & Tendencia",
        "〰  Suavizamiento",
        "🔬  Estacionariedad",
        "🤖  Modelo SARIMA",
        "📋  Datos",
    ])

    # ── TAB 1: Serie & Regresión ──────────────────────────────────────────────
    with t1:
        ts_num = ts.copy()
        ts_num["t"] = np.arange(len(ts_num))
        X = sm.add_constant(ts_num["t"])
        ols = sm.OLS(y.values, X).fit()
        trend = ols.predict(X)

        fig = base_fig(title=f"{flow_name} — Serie histórica & Tendencia OLS")
        fig.add_trace(go.Scatter(x=y.index, y=y.values, name="Original",
                                  line=dict(color=color_main, width=1.8)))
        fig.add_trace(go.Scatter(x=y.index, y=trend, name="Tendencia lineal",
                                  line=dict(color=C["red"], width=2, dash="dash")))
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**Estadísticos descriptivos**")
            desc = y.describe().rename("Trade_Value").to_frame()
            st.dataframe(desc.style.format("${:,.0f}"), width="stretch")
        with col_b:
            st.markdown("**Distribución**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=y.values, nbinsx=25,
                                             marker_color=color_main, opacity=0.7,
                                             name="Frecuencia", histnorm="probability density"))
            fig_hist.update_layout(**LAYOUT_BASE, title="Histograma",
                                    xaxis_tickprefix="$", xaxis_tickformat=",.0f")
            st.plotly_chart(fig_hist, width="stretch")

    # ── TAB 2: Suavizamiento ─────────────────────────────────────────────────
    with t2:
        col1, col2 = st.columns([3, 1])
        with col2:
            window = st.slider("Ventana MA (meses)", 2, 24, 4)
            forecast_m = st.slider("Meses de pronóstico", 6, 60, 24)

        ma = y.rolling(window=window).mean()
        exp_m = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()
        last_date = y.index[-1]
        fut_idx = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                periods=forecast_m, freq="MS")
        fc_vals = exp_m.forecast(forecast_m)

        with col1:
            fig = base_fig(title=f"Suavizamiento — MA({window}) & Exp. Smoothing + {forecast_m}m forecast")
            fig.add_trace(go.Scatter(x=y.index, y=y.values, name="Original",
                                      line=dict(color=color_main, width=1.5, dash="dot"), opacity=0.6))
            fig.add_trace(go.Scatter(x=y.index, y=ma.values, name=f"MA({window})",
                                      line=dict(color=C["cyan"], width=2)))
            fig.add_trace(go.Scatter(x=y.index, y=exp_m.fittedvalues.values,
                                      name="Exp. Smooth (fit)", line=dict(color=C["green"], width=2)))
            fig.add_trace(go.Scatter(x=fut_idx, y=fc_vals.values,
                                      name="Forecast", line=dict(color=C["purple"], width=2.5, dash="dash")))
            fig.add_vrect(x0=str(last_date), x1=str(fut_idx[-1]),
                          fillcolor=C["purple"], opacity=0.05, layer="below", line_width=0)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, width="stretch")

    # ── TAB 3: ADF + ACF/PACF ───────────────────────────────────────────────
    with t3:
        df_diff = y.diff().dropna()

        adf0 = adfuller(y, autolag="AIC")
        adf1 = adfuller(df_diff, autolag="AIC")

        col_a, col_b = st.columns(2)
        with col_a:
            pval0 = adf0[1]
            est0  = "✅ Estacionaria" if pval0 < 0.05 else "⚠️ No estacionaria"
            st.metric("ADF — Serie original", f"p = {pval0:.4f}", est0)
        with col_b:
            pval1 = adf1[1]
            est1  = "✅ Estacionaria" if pval1 < 0.05 else "⚠️ No estacionaria"
            st.metric("ADF — 1ª diferencia", f"p = {pval1:.4f}", est1)

        st.markdown("---")
        # ACF / PACF manual con plotly
        from statsmodels.tsa.stattools import acf, pacf
        nlags = 30
        acf_vals  = acf(df_diff, nlags=nlags, fft=True)
        pacf_vals = pacf(df_diff, nlags=nlags)
        lags = list(range(nlags + 1))
        ci = 1.96 / np.sqrt(len(df_diff))

        fig_ap = make_subplots(rows=1, cols=2, subplot_titles=["ACF (1ª diferencia)", "PACF (1ª diferencia)"])
        for col_idx, (vals, title) in enumerate([(acf_vals, "ACF"), (pacf_vals, "PACF")], 1):
            for i, v in enumerate(vals):
                fig_ap.add_trace(go.Scatter(x=[i, i], y=[0, v],
                                             mode="lines", line=dict(color=color_main, width=2),
                                             showlegend=False), row=1, col=col_idx)
                fig_ap.add_trace(go.Scatter(x=[i], y=[v], mode="markers",
                                             marker=dict(color=color_main, size=6),
                                             showlegend=False), row=1, col=col_idx)
            fig_ap.add_hline(y=ci,  line=dict(color=C["muted"], dash="dash", width=1), row=1, col=col_idx)
            fig_ap.add_hline(y=-ci, line=dict(color=C["muted"], dash="dash", width=1), row=1, col=col_idx)
            fig_ap.add_hline(y=0,   line=dict(color=C["border"], width=1),             row=1, col=col_idx)
        fig_ap.update_layout(**LAYOUT_BASE, title="Autocorrelación — 1ª diferencia")
        st.plotly_chart(fig_ap, width="stretch")

    # ── TAB 4: SARIMA ────────────────────────────────────────────────────────
    with t4:
        with st.expander("⚙️  Parámetros del modelo", expanded=True):
            cc = st.columns(6)
            p = cc[0].number_input("p", 0, 5, default_order[0])
            d = cc[1].number_input("d", 0, 2, default_order[1])
            q = cc[2].number_input("q", 0, 5, default_order[2])
            P = cc[3].number_input("P", 0, 2, default_seasonal[0])
            D = cc[4].number_input("D", 0, 2, default_seasonal[1])
            Q = cc[5].number_input("Q", 0, 2, default_seasonal[2])
            rc1, rc2 = st.columns(2)
            fc_steps  = rc1.slider("Meses de pronóstico", 6, 60, 24, key=f"fc_{flow_name}")
            ic_level  = rc2.select_slider(
                "Nivel de confianza IC",
                options=[80, 85, 90, 95, 99],
                value=95,
                key=f"ic_{flow_name}",
                format_func=lambda x: f"{x}%"
            )
            alpha = 1 - ic_level / 100

        with st.spinner("Entrenando modelo SARIMA…"):
            y_train = y.iloc[:-12]
            y_test  = y.iloc[-12:]

            # Train/test eval
            try:
                m_train = SARIMAX(y_train, order=(p,d,q), seasonal_order=(P,D,Q,12),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                pred    = m_train.get_forecast(steps=12).predicted_mean
                pred.index = y_test.index
                rmse = np.sqrt(mean_squared_error(y_test, pred))

                # ARIMA benchmark
                arima_m  = ARIMA(y_train, order=(5,1,1)).fit()
                arima_p  = arima_m.get_forecast(steps=12).predicted_mean
                arima_p.index = y_test.index
                arima_rmse = np.sqrt(mean_squared_error(y_test, arima_p))

                # Full model
                m_full  = SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,12),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                fut_idx2 = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1),
                                         periods=fc_steps, freq="MS")
                fc_obj  = m_full.get_forecast(steps=fc_steps)
                fc_mean = fc_obj.predicted_mean
                fc_ci   = fc_obj.conf_int(alpha=alpha)

                # Métricas
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("RMSE SARIMA (test 12m)", fmt_usd(rmse))
                mc2.metric("RMSE ARIMA base (test 12m)", fmt_usd(arima_rmse))
                mc3.metric("Mejora vs ARIMA", fmt_usd(arima_rmse - rmse),
                            delta_color="normal" if arima_rmse > rmse else "inverse")

                # Gráfica
                fig_s = base_fig(title=f"SARIMA({p},{d},{q})×({P},{D},{Q},12) — Fitted & Forecast {fc_steps}m")
                fig_s.add_trace(go.Scatter(x=y.index, y=y.values, name="Original",
                                            line=dict(color=color_main, width=1.8)))
                fig_s.add_trace(go.Scatter(x=m_full.fittedvalues.index,
                                            y=m_full.fittedvalues.values,
                                            name="Fitted", line=dict(color=C["cyan"], width=1.5, dash="dot")))
                fig_s.add_trace(go.Scatter(x=fut_idx2, y=fc_mean.values, name="Forecast",
                                            line=dict(color=C["purple"], width=2.5)))
                fig_s.add_traces([
                    go.Scatter(x=list(fut_idx2) + list(fut_idx2[::-1]),
                               y=list(fc_ci.iloc[:,1]) + list(fc_ci.iloc[:,0])[::-1],
                               fill="toself", fillcolor=f"rgba(188,140,255,0.12)",
                               line=dict(color="rgba(0,0,0,0)"), name=f"IC {ic_level}%")
                ])
                # Test period
                fig_s.add_traces([
                    go.Scatter(x=y_test.index, y=y_test.values, name="Test real",
                               line=dict(color=C["green"], width=2)),
                    go.Scatter(x=pred.index, y=pred.values, name="Test pred",
                               line=dict(color=C["orange"], width=2, dash="dash")),
                ])
                fig_s.add_vrect(x0=str(y.index[-1]), x1=str(fut_idx2[-1]),
                                fillcolor=C["purple"], opacity=0.04, layer="below", line_width=0)
                fig_s.update_yaxes(tickprefix="$", tickformat=",.0f")
                st.plotly_chart(fig_s, width="stretch")

                # Eval table
                df_eval = pd.DataFrame({
                    "Real ($)":     y_test.values,
                    "Predicho ($)": pred.values,
                    "Error abs ($)": (y_test.values - pred.values).__abs__(),
                    "Error (%)":    (abs(y_test.values - pred.values) / y_test.values * 100).round(2),
                }, index=y_test.index.strftime("%Y-%m"))
                st.dataframe(df_eval.style.format({
                    "Real ($)":     "${:,.0f}",
                    "Predicho ($)": "${:,.0f}",
                    "Error abs ($)":"${:,.0f}",
                    "Error (%)":    "{:.2f}%",
                }), width="stretch")

            except Exception as e:
                st.error(f"Error al entrenar el modelo: {e}")

    # ── TAB 5: Datos crudos ──────────────────────────────────────────────────
    with t5:
        show_df = ts.reset_index()[["Month", "Trade_Value"]].copy()
        show_df.columns = ["Mes", "Trade Value (USD)"]
        st.dataframe(show_df.style.format({"Trade Value (USD)": "${:,.0f}"}),
                     width="stretch", height=450)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

if page == "Exportaciones":
    render_ts_module("Exportaciones", (1,1,1), (0,1,1))

elif page == "Importaciones":
    render_ts_module("Importaciones", (4,1,0), (1,1,0))

# ══════════════════════════════════════════════════════════════════════════════
#  MODULE: CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Clusters Municipios":
    st.title("🗺️  Clusters de Municipios")
    st.markdown(f"<p style='color:{C['muted']};margin-top:-.5rem'>Segmentación K-Means · 78 municipios · 2006–2025</p>", unsafe_allow_html=True)

    panel = load_panel()

    # ── Build features ────────────────────────────────────────────────────────
    agg = (panel.groupby(["Municipio_ID","Municipio_NAME","State"], as_index=False)
                .agg(trade_total=("Trade_Value","sum"),
                     trade_mean =("Trade_Value","mean"),
                     trade_std  =("Trade_Value","std"),
                     years      =("Year","nunique")))
    agg["trade_std"] = agg["trade_std"].fillna(0)
    agg["trade_cv"]  = np.where(agg["trade_mean"]>0,
                                agg["trade_std"]/agg["trade_mean"], 0)

    pivot = (panel.pivot_table(index=["Municipio_ID","Municipio_NAME"],
                                columns="Flow", values="Trade_Value",
                                aggfunc="sum", fill_value=0).reset_index())
    for c in ["Imports","Exports"]:
        if c not in pivot.columns: pivot[c] = 0
    pivot = pivot.rename(columns={"Imports":"imports_total","Exports":"exports_total"})

    cdf = agg.merge(pivot, on=["Municipio_ID","Municipio_NAME"], how="left")
    den = cdf["imports_total"] + cdf["exports_total"]
    cdf["import_share"] = np.where(den>0, cdf["imports_total"]/den, 0)
    cdf["export_share"] = np.where(den>0, cdf["exports_total"]/den, 0)

    FEAT = ["trade_total","trade_mean","trade_std","trade_cv","import_share","export_share","years"]
    X = cdf[FEAT].copy()
    for c in ["trade_total","trade_mean","trade_std"]:
        X[c] = np.clip(X[c], 0, X[c].quantile(0.99))
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    # ── Controles K-Means en línea ────────────────────────────────────────────
    with st.expander("⚙️  Parámetros K-Means", expanded=False):
        kc1, kc2 = st.columns(2)
        k_mode   = kc1.radio("Selección de K", ["Automático (Silhouette)", "Manual"])
        k_manual = kc2.slider("K manual", 2, 10, 4)

    # Silhouette calculation
    ks = list(range(2, 11))
    with st.spinner("Calculando métricas de clustering…"):
        inertias, silhs = [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            lb = km.fit_predict(X_sc)
            inertias.append(km.inertia_)
            silhs.append(silhouette_score(X_sc, lb))

    best_k = ks[int(np.argmax(silhs))]
    K_FINAL = best_k if k_mode == "Automático (Silhouette)" else k_manual

    kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=50)
    cdf["cluster"] = kmeans.fit_predict(X_sc)
    cluster_labels = {i: f"Cluster {i}" for i in range(K_FINAL)}

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X_sc)
    cdf["PC1"] = Z[:,0]
    cdf["PC2"] = Z[:,1]

    PALETTE = px.colors.qualitative.Bold

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📐  Elbow & Silhouette",
        "🔵  Mapa PCA",
        "📊  Perfil de Clusters",
        "🏆  Top Municipios",
    ])

    with tab1:
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("K seleccionado", K_FINAL)
        kpi2.metric("Silhouette score", f"{silhs[K_FINAL-2]:.3f}")

        fig_el = make_subplots(rows=1, cols=2,
                               subplot_titles=["Elbow (Inercia)", "Silhouette Score"])
        fig_el.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                                     line=dict(color=C["blue"], width=2),
                                     marker=dict(size=8, color=C["blue"])), row=1, col=1)
        fig_el.add_trace(go.Scatter(x=ks, y=silhs, mode="lines+markers",
                                     line=dict(color=C["green"], width=2),
                                     marker=dict(size=8, color=C["green"])), row=1, col=2)
        for col_idx in [1, 2]:
            fig_el.add_vline(x=K_FINAL, line=dict(color=C["red"], dash="dash", width=1.5),
                              row=1, col=col_idx)
        fig_el.update_layout(**LAYOUT_BASE, showlegend=False)
        st.plotly_chart(fig_el, width="stretch")

    with tab2:
        var_exp = pca.explained_variance_ratio_.sum()
        fig_pca = go.Figure()
        for cl in sorted(cdf["cluster"].unique()):
            sub = cdf[cdf["cluster"]==cl]
            fig_pca.add_trace(go.Scatter(
                x=sub["PC1"], y=sub["PC2"],
                mode="markers+text",
                text=sub["Municipio_NAME"],
                textfont=dict(size=8, color=PALETTE[cl % len(PALETTE)]),
                textposition="top center",
                marker=dict(size=10, color=PALETTE[cl % len(PALETTE)],
                            line=dict(color=C["bg"], width=1)),
                name=f"Cluster {cl}",
                customdata=sub[["Municipio_NAME","State","trade_total","import_share","export_share"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Trade total: $%{customdata[2]:,.0f}<br>"
                    "Import share: %{customdata[3]:.1%}<br>"
                    "Export share: %{customdata[4]:.1%}<extra></extra>"
                )
            ))
        fig_pca.update_layout(**LAYOUT_BASE,
                               title=f"PCA 2D — K={K_FINAL} clusters  |  varianza explicada: {var_exp:.1%}",
                               xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig_pca, width="stretch")

    with tab3:
        profile = cdf.groupby("cluster")[FEAT].mean().round(2)

        # Radar normalizado
        prof_norm = profile.copy()
        for col in FEAT:
            mn, mx = prof_norm[col].min(), prof_norm[col].max()
            prof_norm[col] = (prof_norm[col] - mn) / (mx - mn + 1e-9)

        fig_radar = go.Figure()
        for cl in sorted(cdf["cluster"].unique()):
            vals = prof_norm.loc[cl].tolist()
            vals += [vals[0]]
            cats = FEAT + [FEAT[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself", name=f"Cluster {cl}",
                line=dict(color=PALETTE[cl % len(PALETTE)], width=2),
                fillcolor=PALETTE[cl % len(PALETTE)].replace("rgb", "rgba").replace(")", ",0.10)") if "rgb" in PALETTE[cl % len(PALETTE)] else PALETTE[cl % len(PALETTE)],
            ))
        fig_radar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans", color=C["text"]),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, showticklabels=False, gridcolor=C["border"]),
                angularaxis=dict(gridcolor=C["border"], tickfont=dict(size=10)),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title="Perfil normalizado por cluster",
            margin=dict(l=60, r=60, t=60, b=40),
        )
        st.plotly_chart(fig_radar, width="stretch")

        # Tabla de medias
        st.markdown("**Medias por cluster**")
        st.dataframe(profile.style.format({
            "trade_total": "${:,.0f}",
            "trade_mean":  "${:,.0f}",
            "trade_std":   "${:,.0f}",
            "trade_cv":    "{:.3f}",
            "import_share":"{:.2%}",
            "export_share":"{:.2%}",
            "years":       "{:.0f}",
        }).background_gradient(cmap="Blues"), width="stretch")

    with tab4:
        sel_cl = st.selectbox("Filtrar por cluster", ["Todos"] + [f"Cluster {i}" for i in sorted(cdf["cluster"].unique())])
        top_df = cdf.copy()
        if sel_cl != "Todos":
            top_df = top_df[top_df["cluster"] == int(sel_cl.split()[-1])]
        top_df = top_df.sort_values("trade_total", ascending=False)

        show_cols = ["Municipio_NAME","State","cluster","trade_total",
                     "imports_total","exports_total","import_share","export_share","trade_cv"]
        st.dataframe(top_df[show_cols].style.format({
            "trade_total":    "${:,.0f}",
            "imports_total":  "${:,.0f}",
            "exports_total":  "${:,.0f}",
            "import_share":   "{:.1%}",
            "export_share":   "{:.1%}",
            "trade_cv":       "{:.3f}",
        }), width="stretch", height=480)

        # Treemap
        fig_tree = px.treemap(
            top_df, path=["State","Municipio_NAME"],
            values="trade_total", color="cluster",
            color_discrete_sequence=PALETTE,
            title="Treemap: Trade Total por Estado y Municipio",
        )
        fig_tree.update_layout(**{k:v for k,v in LAYOUT_BASE.items()
                                   if k not in ["xaxis","yaxis","hovermode"]})
        fig_tree.update_traces(textinfo="label+percent root")
        st.plotly_chart(fig_tree, width="stretch")
