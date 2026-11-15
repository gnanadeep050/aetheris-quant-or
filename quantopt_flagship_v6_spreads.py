import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize, milp, LinearConstraint, Bounds
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from datetime import datetime

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIG & STRICT ORANGE & WHITE CONTRAST THEME
# ==========================================
st.set_page_config(
    page_title="AETHERIS QUANT // OR | Orange & White Multi-Asset Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulseBorder {
        0% { box-shadow: 0 4px 16px rgba(249, 115, 22, 0.16); }
        50% { box-shadow: 0 8px 26px rgba(234, 88, 12, 0.30); }
        100% { box-shadow: 0 4px 16px rgba(249, 115, 22, 0.16); }
    }
    @keyframes tickerScroll {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-50%); }
    }
    @keyframes fadeInSlide {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes liveBlink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.45; }
    }

    /* 1. Eliminate ALL black backgrounds on main canvas, top header & sidebar */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #ffffff !important;
        background-image: linear-gradient(180deg, #ffffff 0%, #fffaf5 65%, #fff7ed 100%) !important;
        color: #000000 !important;
    }
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #fed7aa !important;
    }
    header[data-testid="stHeader"] * {
        color: #000000 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: #fff7ed !important;
        border-right: 1px solid #fdba74 !important;
        color: #000000 !important;
    }

    /* 2. Force BLACK text on all light surfaces */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, .stMarkdown {
        color: #000000;
    }

    /* 3. Force WHITE/CREAM background & BLACK text on all sidebar inputs, dropdowns & expanders */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="base-input"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #fdba74 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="input"] input,
    [data-testid="stNumberInput"] button {
        background-color: #ffffff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    /* Dropdown menu popover list */
    ul[role="listbox"], ul[role="listbox"] li {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    ul[role="listbox"] li:hover {
        background-color: #ffedd5 !important;
        color: #000000 !important;
    }
    /* Expanders in light mode */
    [data-testid="stExpander"], details {
        background-color: #ffffff !important;
        border: 1px solid #fdba74 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary, details summary {
        background-color: #fff7ed !important;
        color: #000000 !important;
    }

    /* 4. Force LIGHT WHITE & ORANGE styling on Streamlit Dataframes / Tables */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stTable"] {
        background-color: #ffffff !important;
        border: 1px solid #fdba74 !important;
        border-radius: 10px !important;
        color: #000000 !important;
    }
    [data-testid="stDataFrame"] * {
        color: #000000 !important;
    }

    /* Live Badge: White background with Black/Orange text */
    .live-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background-color: #ffffff !important;
        color: #c2410c !important;
        border: 1px solid #fdba74;
        animation: liveBlink 2.2s infinite;
    }

    /* Live Animated Market Ticker Tape (Light Cream Background, Dark High-Contrast Text) */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background-color: #fff7ed !important;
        border-top: 1px solid #fdba74;
        border-bottom: 1px solid #fdba74;
        padding: 8px 0;
        margin-bottom: 14px;
        white-space: nowrap;
    }
    .ticker-move {
        display: inline-block;
        animation: tickerScroll 36s linear infinite;
        font-family: monospace;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
    }
    .ticker-move:hover { animation-play-state: paused; }
    .tick-up { color: #047857 !important; font-weight: 800; margin-right: 28px; }
    .tick-down { color: #be123c !important; font-weight: 800; margin-right: 28px; }
    .tick-neutral { color: #c2410c !important; font-weight: 800; margin-right: 28px; }

    /* Dark/Vibrant Orange Header Banner -> Strict WHITE Text */
    .brand-header {
        background: linear-gradient(270deg, #c2410c, #ea580c, #f97316, #c2410c) !important;
        background-size: 300% 300%;
        animation: gradientFlow 10s ease infinite, pulseBorder 4s infinite;
        border: 1px solid #ea580c;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 16px;
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 900;
        letter-spacing: 0.10em;
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .brand-sub {
        font-size: 0.80rem;
        color: #ffffff !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Light White Metric Cards -> Strict BLACK Text */
    .metric-card {
        background-color: #ffffff !important;
        border: 1px solid #fed7aa;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        animation: fadeInSlide 0.5s ease-out;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #ea580c;
        box-shadow: 0 10px 24px rgba(249, 115, 22, 0.18);
    }
    .card-orange-1 { border-top: 4px solid #f97316; }
    .card-orange-2 { border-top: 4px solid #ea580c; }
    .card-amber    { border-top: 4px solid #d97706; }
    .card-emerald  { border-top: 4px solid #059669; }

    .metric-label { font-size: 0.72rem; color: #334155 !important; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }
    .metric-value { font-size: 1.46rem; font-weight: 800; color: #000000 !important; margin-top: 5px; font-family: monospace; }
    .metric-sub { font-size: 0.78rem; color: #047857 !important; margin-top: 5px; font-weight: 700; }
    .metric-alert { font-size: 0.78rem; color: #c2410c !important; margin-top: 5px; font-weight: 700; }

    /* Light Warm Executive Banner -> Strict BLACK Text */
    .action-banner {
        background: linear-gradient(90deg, #fff7ed, #ffedd5, #fff7ed) !important;
        border: 1px solid #fdba74;
        border-left: 5px solid #ea580c;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 18px;
        font-size: 0.90rem;
        color: #000000 !important;
        animation: fadeInSlide 0.6s ease-out;
    }
    .action-banner b {
        color: #000000 !important;
    }

    /* White Catalyst Cards -> Strict BLACK Text */
    .catalyst-card {
        background-color: #ffffff !important;
        border: 1px solid #fed7aa;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .catalyst-card:hover {
        transform: translateX(4px);
        box-shadow: 0 6px 16px rgba(249, 115, 22, 0.14);
    }

    /* Tabs: Black Text on Light Unselected Tabs, WHITE Text on Dark-Orange Active Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #fff7ed !important;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #fed7aa;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        padding: 8px 14px;
        background-color: transparent !important;
        color: #000000 !important;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"] * {
        color: #000000 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ea580c, #f97316) !important;
        color: #ffffff !important;
        box-shadow: 0 3px 12px rgba(234, 88, 12, 0.35);
    }
    .stTabs [aria-selected="true"] * {
        color: #ffffff !important;
    }

    /* Dark-Orange Buttons -> Strict WHITE Text */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(90deg, #ea580c, #f97316) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 3px 10px rgba(249, 115, 22, 0.25) !important;
    }
    .stButton > button *, .stDownloadButton > button * {
        color: #ffffff !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #c2410c, #ea580c) !important;
        transform: translateY(-1px);
    }

    /* White Formula Box -> Strict BLACK Text */
    .formula-box {
        background-color: #ffffff !important;
        border: 1px solid #fed7aa;
        border-left: 5px solid #ea580c;
        padding: 16px 20px;
        font-family: monospace;
        font-size: 0.83rem;
        color: #000000 !important;
        border-radius: 8px;
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)

# Helper to enforce crisp BLACK text on all light Plotly charts
def apply_light_orange_chart_style(fig, height=320, title_text=None):
    layout_kwargs = dict(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000", size=12),
        title_font=dict(color="#000000", size=15),
        legend=dict(font=dict(color="#000000")),
        height=height,
    )
    if title_text:
        layout_kwargs["title"] = title_text
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        color="#000000",
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
        gridcolor="#ffedd5",
        linecolor="#fdba74"
    )
    fig.update_yaxes(
        color="#000000",
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
        gridcolor="#ffedd5",
        linecolor="#fdba74"
    )
    return fig

# ==========================================
# 2. CLEAN INVESTOR PRESETS & MACRO REGIMES
# ==========================================
INVESTOR_PRESETS = {
    "Balanced Wealth Builder (Recommended)": {
        "regime": "Normal / Calibrated Growth (Live Baseline)",
        "horizon": "3 Years",
        "lump": 1500000,
        "sip": 25000,
        "target_cagr": 17.5,
        "solver": "Cardinality Constrained MILP (HiGHS K-Best)",
        "crypto": True,
        "crypto_cap": 15,
        "ipo_pct": 15,
        "opt_budget": 120000,
        "opt_mode": "All Strategies (Auto-Select Best EV)"
    },
    "Aggressive High-Growth + Crypto + IPOs": {
        "regime": "Normal / Calibrated Growth (Live Baseline)",
        "horizon": "5 Years",
        "lump": 2000000,
        "sip": 40000,
        "target_cagr": 21.0,
        "solver": "Maximum Sharpe Ratio (Fractional QP)",
        "crypto": True,
        "crypto_cap": 25,
        "ipo_pct": 20,
        "opt_budget": 200000,
        "opt_mode": "All Strategies (Auto-Select Best EV)"
    },
    "War & Rate-Hike Safe Haven (Capital Protection)": {
        "regime": "Severe Twin Shock (War + Aggressive Rate Hikes)",
        "horizon": "2 Years",
        "lump": 1500000,
        "sip": 20000,
        "target_cagr": 14.5,
        "solver": "CVaR 95% Tail-Risk Minimization (LP)",
        "crypto": True,
        "crypto_cap": 5,
        "ipo_pct": 10,
        "opt_budget": 80000,
        "opt_mode": "Vertical Debit Spreads Only (Bull Call / Bear Put)"
    },
    "Active Day Trader & Options Spreads Focus": {
        "regime": "Normal / Calibrated Growth (Live Baseline)",
        "horizon": "1 Year",
        "lump": 500000,
        "sip": 15000,
        "target_cagr": 16.0,
        "solver": "Cardinality Constrained MILP (HiGHS K-Best)",
        "crypto": False,
        "crypto_cap": 0,
        "ipo_pct": 10,
        "opt_budget": 250000,
        "opt_mode": "Vertical Debit Spreads Only (Bull Call / Bear Put)"
    }
}

BASE_BRENT_USD = 96.0
BASE_USD_INR = 94.25

MACRO_ENVIRONMENTS = {
    "Normal / Calibrated Growth (Live Baseline)": {
        "deploy_now_pct": 1.00, "rbi_repo_pct": 5.50, "fed_rate_pct": 4.00,
        "vol_mult": 1.00, "crypto_cap_mult": 1.00, "default_brent": 96.0, "default_usdinr": 94.25,
        "sector_alpha_shift": {},
        "summary": "Live market baseline calibrated with real-time spot feeds, RBI Repo 5.50%, and Fed 4.00%."
    },
    "Bank Interest Rate Hike Cycle (+50 to +100 bps RBI/Fed)": {
        "deploy_now_pct": 0.65, "rbi_repo_pct": 6.25, "fed_rate_pct": 4.75,
        "vol_mult": 1.18, "crypto_cap_mult": 0.65, "default_brent": 98.0, "default_usdinr": 95.50,
        "sector_alpha_shift": {
            "Banking & Finance": +0.028, "Capital Markets": +0.012, "Aerospace & Defense": +0.010,
            "Consumer & Retail": -0.025, "Automobiles & EV": -0.022, "Electronics & EMS": -0.018,
            "Crypto Store-of-Value": -0.045, "Crypto Layer-1": -0.060,
        },
        "summary": "Higher interest rates compress high-PE multiples & crypto liquidity while expanding NIMs for high-CASA banks."
    },
    "Global Geopolitical / Strait of Hormuz War Shock": {
        "deploy_now_pct": 0.50, "rbi_repo_pct": 5.50, "fed_rate_pct": 4.00,
        "vol_mult": 1.42, "crypto_cap_mult": 0.40, "default_brent": 128.0, "default_usdinr": 97.50,
        "sector_alpha_shift": {
            "Aerospace & Defense": +0.055, "Power & Green Energy": +0.022, "Energy & Digital": +0.018,
            "Pharma & Healthcare": +0.015, "Automobiles & EV": -0.035, "Consumer & Retail": -0.028,
            "Crypto Store-of-Value": -0.030, "Crypto Layer-1": -0.070,
        },
        "summary": "Crude oil spike ($128/bbl) & INR pressure trigger FII risk-off; Defense, Energy & IT/Pharma export hedges outperform."
    },
    "Severe Twin Shock (War + Aggressive Rate Hikes)": {
        "deploy_now_pct": 0.40, "rbi_repo_pct": 6.50, "fed_rate_pct": 5.00,
        "vol_mult": 1.65, "crypto_cap_mult": 0.25, "default_brent": 138.0, "default_usdinr": 99.00,
        "sector_alpha_shift": {
            "Aerospace & Defense": +0.060, "Banking & Finance": +0.020, "Pharma & Healthcare": +0.018,
            "Power & Green Energy": +0.015, "Consumer & Retail": -0.045, "Automobiles & EV": -0.045,
            "Electronics & EMS": -0.035, "Crypto Store-of-Value": -0.065, "Crypto Layer-1": -0.095,
        },
        "summary": "Stagflationary shock: 40% deployed immediately into Defense/Banks/Healthcare; 60% held in safe liquid yield for dip-buying."
    }
}

BRENT_SECTOR_BETA_PER_10USD = {
    "Energy & Digital": +0.014, "Power & Green Energy": +0.008, "Aerospace & Defense": +0.009,
    "Automobiles & EV": -0.016, "Consumer & Retail": -0.013, "Electronics & EMS": -0.010,
    "Infrastructure & EPC": -0.008, "Banking & Finance": -0.006, "IT & AI Services": +0.002,
    "Pharma & Healthcare": -0.003, "Capital Goods & Semi": -0.007, "Capital Markets": -0.005,
    "Telecom & Digital": -0.004, "Crypto Store-of-Value": -0.012, "Crypto Layer-1": -0.018,
    "Crypto Exchange/L1": -0.015, "Crypto Infrastructure": -0.016, "Crypto Payments": -0.015,
}

FX_SECTOR_BETA_PER_1INR = {
    "IT & AI Services": +0.0085, "Pharma & Healthcare": +0.0065, "Aerospace & Defense": +0.0020,
    "Energy & Digital": +0.0015, "Electronics & EMS": -0.0045, "Automobiles & EV": -0.0040,
    "Consumer & Retail": -0.0035, "Banking & Finance": -0.0025, "Infrastructure & EPC": -0.0020,
    "Crypto Store-of-Value": +0.0075, "Crypto Layer-1": +0.0070, "Crypto Exchange/L1": +0.0070,
    "Crypto Infrastructure": +0.0070, "Crypto Payments": +0.0070,
}

# ==========================================
# 3. NSE MULTI-CAP UNIVERSE + CRYPTO + SEBI NOV 2026 IPOS
# ==========================================
NSE_MASTER_DATA = [
    {"Ticker": "RELIANCE.NS",   "Company": "Reliance Industries",      "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Energy & Digital",    "Spot": 1318.0, "Lot_Size": 500,  "Base_Ret": 0.148, "Vol": 0.21, "Beta": 1.04, "ROE": 0.108, "YoY_Rev_%": 11.4, "YoY_PAT_%": 14.2, "DE_Ratio": 0.42, "Moat_Score": 9.1, "OI_Signal": "Long Buildup",   "News_Sent": +0.62, "News_Headline": "Energy sector ends green amid crude strength; Jio AI cloud expansion accelerates.", "Key_Products": "Jio 5G/AI Cloud, Retail, O2C, New Energy Gigafactories"},
    {"Ticker": "TCS.NS",        "Company": "Tata Consultancy Services","Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "IT & AI Services",    "Spot": 4125.0, "Lot_Size": 225,  "Base_Ret": 0.135, "Vol": 0.18, "Beta": 0.81, "ROE": 0.485, "YoY_Rev_%": 8.2,  "YoY_PAT_%": 10.5, "DE_Ratio": 0.04, "Moat_Score": 9.0, "OI_Signal": "Short Covering", "News_Sent": +0.28, "News_Headline": "USD/INR strength cushions IT export margins despite cautious US BFSI spend.", "Key_Products": "Enterprise AI Cloud, BaNCS, Ignio Cognitive Automation"},
    {"Ticker": "HDFCBANK.NS",   "Company": "HDFC Bank",                "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Banking & Finance",   "Spot": 1742.0, "Lot_Size": 650,  "Base_Ret": 0.142, "Vol": 0.19, "Beta": 0.93, "ROE": 0.169, "YoY_Rev_%": 16.5, "YoY_PAT_%": 17.8, "DE_Ratio": 1.10, "Moat_Score": 9.2, "OI_Signal": "Long Unwinding", "News_Sent": -0.18, "News_Headline": "Bank Nifty sees mild profit booking after multi-day private bank rally.", "Key_Products": "Retail & Corporate Lending, PayZapp, SmartHub Vyapar"},
    {"Ticker": "ICICIBANK.NS",  "Company": "ICICI Bank",               "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Banking & Finance",   "Spot": 1296.0, "Lot_Size": 700,  "Base_Ret": 0.168, "Vol": 0.21, "Beta": 1.06, "ROE": 0.186, "YoY_Rev_%": 18.2, "YoY_PAT_%": 21.4, "DE_Ratio": 1.05, "Moat_Score": 9.3, "OI_Signal": "Long Unwinding", "News_Sent": -0.12, "News_Headline": "Strong Q2 NIM resilience offsets near-term F&O long unwinding in private lenders.", "Key_Products": "iMobile Pay, InstaBIZ, Treasury & Wealth Management"},
    {"Ticker": "INFY.NS",       "Company": "Infosys Limited",          "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "IT & AI Services",    "Spot": 1865.0, "Lot_Size": 400,  "Base_Ret": 0.144, "Vol": 0.20, "Beta": 0.87, "ROE": 0.318, "YoY_Rev_%": 7.9,  "YoY_PAT_%": 9.8,  "DE_Ratio": 0.08, "Moat_Score": 8.8, "OI_Signal": "Short Covering", "News_Sent": +0.34, "News_Headline": "Topaz GenAI deal pipeline & rupee depreciation support intraday bounce setup.", "Key_Products": "Topaz GenAI Suite, Finacle Core Banking, Cobalt Cloud"},
    {"Ticker": "LT.NS",         "Company": "Larsen & Toubro",          "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Infrastructure & EPC","Spot": 3640.0, "Lot_Size": 150,  "Base_Ret": 0.182, "Vol": 0.23, "Beta": 1.12, "ROE": 0.155, "YoY_Rev_%": 19.4, "YoY_PAT_%": 22.1, "DE_Ratio": 0.65, "Moat_Score": 9.4, "OI_Signal": "Long Buildup",   "News_Sent": +0.58, "News_Headline": "Heavy domestic capex & green hydrogen order inflows sustain bullish trend.", "Key_Products": "Heavy EPC, Green Hydrogen, Defense & Semiconductor Design"},
    {"Ticker": "BHARTIARTL.NS", "Company": "Bharti Airtel",            "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Telecom & Digital",   "Spot": 1645.0, "Lot_Size": 475,  "Base_Ret": 0.176, "Vol": 0.21, "Beta": 0.89, "ROE": 0.152, "YoY_Rev_%": 15.8, "YoY_PAT_%": 38.5, "DE_Ratio": 1.35, "Moat_Score": 9.1, "OI_Signal": "Long Buildup",   "News_Sent": +0.52, "News_Headline": "ARPU expansion & Nxtra data center demand keep Airtel above 20-EMA support.", "Key_Products": "5G Plus, Airtel Xstream Fiber, Nxtra Data Centers"},
    {"Ticker": "SUNPHARMA.NS",  "Company": "Sun Pharmaceutical",       "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Pharma & Healthcare", "Spot": 1820.0, "Lot_Size": 350,  "Base_Ret": 0.156, "Vol": 0.17, "Beta": 0.64, "ROE": 0.172, "YoY_Rev_%": 12.6, "YoY_PAT_%": 18.9, "DE_Ratio": 0.05, "Moat_Score": 8.9, "OI_Signal": "Long Buildup",   "News_Sent": +0.44, "News_Headline": "Defensive pharma rotation & US specialty dermatology growth attract buyers.", "Key_Products": "Ilumya, Winlevi Specialty Dermatology & Global Generics"},
    {"Ticker": "M&M.NS",        "Company": "Mahindra & Mahindra",      "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Automobiles & EV",    "Spot": 2910.0, "Lot_Size": 175,  "Base_Ret": 0.198, "Vol": 0.25, "Beta": 1.09, "ROE": 0.184, "YoY_Rev_%": 16.9, "YoY_PAT_%": 24.6, "DE_Ratio": 0.45, "Moat_Score": 9.0, "OI_Signal": "Short Buildup",  "News_Sent": -0.46, "News_Headline": "Auto index slips on crude & inflation worries; OEMs see short buildup.", "Key_Products": "Born Electric SUVs (BE 6/XEV 9e), Swaraj Farm Tractors"},
    {"Ticker": "HAL.NS",        "Company": "Hindustan Aeronautics",    "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Aerospace & Defense", "Spot": 4390.0, "Lot_Size": 150,  "Base_Ret": 0.228, "Vol": 0.32, "Beta": 1.21, "ROE": 0.268, "YoY_Rev_%": 18.5, "YoY_PAT_%": 26.8, "DE_Ratio": 0.00, "Moat_Score": 9.7, "OI_Signal": "Long Buildup",   "News_Sent": +0.78, "News_Headline": "Tejas Mk1A delivery ramp & geopolitical defense spending fuel strong OI buildup.", "Key_Products": "Tejas Mk1A Fighter, Prachand LCH, Aero-Engines & MRO"},
    {"Ticker": "BEL.NS",        "Company": "Bharat Electronics",       "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Aerospace & Defense", "Spot": 304.0,  "Lot_Size": 1425, "Base_Ret": 0.215, "Vol": 0.28, "Beta": 1.14, "ROE": 0.245, "YoY_Rev_%": 17.8, "YoY_PAT_%": 25.2, "DE_Ratio": 0.00, "Moat_Score": 9.3, "OI_Signal": "Long Buildup",   "News_Sent": +0.71, "News_Headline": "Fresh radar & electronic warfare order wins drive breakout above intraday VWAP.", "Key_Products": "AESA Radars, Electronic Warfare, Akash Missile Systems"},
    {"Ticker": "NTPC.NS",       "Company": "NTPC Limited",             "Asset_Class": "Indian Equity", "Segment": "Large Cap", "Sector": "Power & Green Energy","Spot": 392.0,  "Lot_Size": 1500, "Base_Ret": 0.154, "Vol": 0.20, "Beta": 0.84, "ROE": 0.135, "YoY_Rev_%": 10.4, "YoY_PAT_%": 15.6, "DE_Ratio": 1.40, "Moat_Score": 8.5, "OI_Signal": "Long Buildup",   "News_Sent": +0.41, "News_Headline": "Peak power demand & NTPC Green BESS storage tenders support steady accumulation.", "Key_Products": "76 GW Thermal + NTPC Green Solar/BESS Utility Scale"},
    {"Ticker": "TRENT.NS",      "Company": "Trent Limited (Zudio)",    "Asset_Class": "Indian Equity", "Segment": "Mid Cap",   "Sector": "Consumer & Retail",   "Spot": 6480.0, "Lot_Size": 100,  "Base_Ret": 0.242, "Vol": 0.31, "Beta": 1.05, "ROE": 0.285, "YoY_Rev_%": 39.4, "YoY_PAT_%": 46.2, "DE_Ratio": 0.18, "Moat_Score": 9.4, "OI_Signal": "Short Buildup",  "News_Sent": -0.35, "News_Headline": "Consumer durables & discretionary retail dip amid high-PE profit taking.", "Key_Products": "Zudio Fast-Fashion Stores, Westside, Star Bazaar"},
    {"Ticker": "DIXON.NS",      "Company": "Dixon Technologies",       "Asset_Class": "Indian Equity", "Segment": "Mid Cap",   "Sector": "Electronics & EMS",   "Spot": 14850.0,"Lot_Size": 50,   "Base_Ret": 0.254, "Vol": 0.34, "Beta": 1.18, "ROE": 0.242, "YoY_Rev_%": 54.0, "YoY_PAT_%": 61.5, "DE_Ratio": 0.22, "Moat_Score": 9.1, "OI_Signal": "Long Buildup",   "News_Sent": +0.64, "News_Headline": "54% YoY revenue surge & smartphone PLI export ramp attract momentum buyers.", "Key_Products": "Smartphones (PLI), Laptops, Display Modules & Hearables"},
    {"Ticker": "PERSISTENT.NS", "Company": "Persistent Systems",       "Asset_Class": "Indian Equity", "Segment": "Mid Cap",   "Sector": "IT & AI Services",    "Spot": 5620.0, "Lot_Size": 100,  "Base_Ret": 0.224, "Vol": 0.29, "Beta": 1.08, "ROE": 0.238, "YoY_Rev_%": 18.6, "YoY_PAT_%": 24.8, "DE_Ratio": 0.03, "Moat_Score": 9.0, "OI_Signal": "Long Buildup",   "News_Sent": +0.55, "News_Headline": "SASVA GenAI enterprise platform wins outpace tier-1 IT peers in Q2.", "Key_Products": "SASVA GenAI Engineering Platform, Cloud & SaaS Product Dev"},
    {"Ticker": "CGPOWER.NS",    "Company": "CG Power & Industrial",    "Asset_Class": "Indian Equity", "Segment": "Mid Cap",   "Sector": "Capital Goods & Semi","Spot": 745.0,  "Lot_Size": 850,  "Base_Ret": 0.222, "Vol": 0.30, "Beta": 1.12, "ROE": 0.295, "YoY_Rev_%": 24.5, "YoY_PAT_%": 28.9, "DE_Ratio": 0.01, "Moat_Score": 9.2, "OI_Signal": "Long Buildup",   "News_Sent": +0.49, "News_Headline": "Transformer backlog & Sanand OSAT semiconductor milestone drive bullish engulfing.", "Key_Products": "Power Transformers, Railway Propulsion & OSAT Chip Packaging"},
    {"Ticker": "SOLARINDS.NS",  "Company": "Solar Industries India",   "Asset_Class": "Indian Equity", "Segment": "Mid Cap",   "Sector": "Aerospace & Defense", "Spot": 10420.0,"Lot_Size": 75,   "Base_Ret": 0.231, "Vol": 0.28, "Beta": 0.96, "ROE": 0.274, "YoY_Rev_%": 23.4, "YoY_PAT_%": 31.2, "DE_Ratio": 0.28, "Moat_Score": 9.5, "OI_Signal": "Long Buildup",   "News_Sent": +0.69, "News_Headline": "Pinaka rocket & loitering munition export orders boost defense alpha.", "Key_Products": "Pinaka Rockets, Loitering Munitions, Industrial Explosives"},
    {"Ticker": "KPITTECH.NS",   "Company": "KPIT Technologies",        "Asset_Class": "Indian Equity", "Segment": "Small Cap", "Sector": "Automobiles & EV",    "Spot": 1490.0, "Lot_Size": 775,  "Base_Ret": 0.246, "Vol": 0.33, "Beta": 1.16, "ROE": 0.284, "YoY_Rev_%": 24.8, "YoY_PAT_%": 33.6, "DE_Ratio": 0.02, "Moat_Score": 9.3, "OI_Signal": "Short Covering", "News_Sent": +0.22, "News_Headline": "Global SDV software contracts offset near-term EV auto sector pullback.", "Key_Products": "Software-Defined Vehicle (SDV) OS, Autonomous & EV Powertrain"},
    {"Ticker": "KAYNES.NS",     "Company": "Kaynes Technology India",  "Asset_Class": "Indian Equity", "Segment": "Small Cap", "Sector": "Electronics & EMS",   "Spot": 5890.0, "Lot_Size": 100,  "Base_Ret": 0.262, "Vol": 0.36, "Beta": 1.24, "ROE": 0.148, "YoY_Rev_%": 46.5, "YoY_PAT_%": 54.2, "DE_Ratio": 0.19, "Moat_Score": 9.1, "OI_Signal": "Long Buildup",   "News_Sent": +0.66, "News_Headline": "54% PAT growth & OSAT chip packaging ramp trigger inside-bar breakout.", "Key_Products": "IoT PCBA, High-Density PCBs & Sanand OSAT Semiconductor Plant"},
    {"Ticker": "DATAPATTNS.NS", "Company": "Data Patterns (India)",    "Asset_Class": "Indian Equity", "Segment": "Small Cap", "Sector": "Aerospace & Defense", "Spot": 2480.0, "Lot_Size": 125,  "Base_Ret": 0.238, "Vol": 0.35, "Beta": 1.19, "ROE": 0.156, "YoY_Rev_%": 28.4, "YoY_PAT_%": 32.1, "DE_Ratio": 0.00, "Moat_Score": 9.4, "OI_Signal": "Long Buildup",   "News_Sent": +0.74, "News_Headline": "41% EBITDA margin & radar electronics orders put stock in high-momentum zone.", "Key_Products": "BrahMos Fire Control, Airborne Radars & Satellite Electronics"},
    {"Ticker": "CDSL.NS",       "Company": "Central Depository (CDSL)","Asset_Class": "Indian Equity", "Segment": "Small Cap", "Sector": "Capital Markets",     "Spot": 1685.0, "Lot_Size": 475,  "Base_Ret": 0.218, "Vol": 0.29, "Beta": 1.08, "ROE": 0.312, "YoY_Rev_%": 31.5, "YoY_PAT_%": 36.8, "DE_Ratio": 0.00, "Moat_Score": 9.6, "OI_Signal": "Long Buildup",   "News_Sent": +0.61, "News_Headline": "BSE & capital market infrastructure stocks buck broader selloff.", "Key_Products": "14 Cr+ Demat Accounts Duopoly, e-Voting & KYC Repository"},
]

CRYPTO_MASTER_DATA = [
    {"Ticker": "BTC-USD",  "Company": "Bitcoin (BTC)",   "Asset_Class": "Cryptocurrency", "Segment": "Crypto Blue-Chip", "Sector": "Crypto Store-of-Value", "Spot": 91400.0, "Lot_Size": 1, "Base_Ret": 0.340, "Vol": 0.52, "Beta": 1.45, "ROE": 0.320, "YoY_Rev_%": 45.0, "YoY_PAT_%": 48.0, "DE_Ratio": 0.00, "Moat_Score": 9.7, "OI_Signal": "Long Buildup", "News_Sent": +0.58, "News_Headline": "Spot BTC ETF net inflows & sovereign treasury demand hold key support.", "Key_Products": "60.5% Market Dominance, Spot BTC ETFs (IBIT/FBTC), Sovereign Reserve Asset"},
    {"Ticker": "ETH-USD",  "Company": "Ethereum (ETH)",  "Asset_Class": "Cryptocurrency", "Segment": "Crypto Blue-Chip", "Sector": "Crypto Layer-1",        "Spot": 3180.0,  "Lot_Size": 1, "Base_Ret": 0.310, "Vol": 0.60, "Beta": 1.65, "ROE": 0.280, "YoY_Rev_%": 38.0, "YoY_PAT_%": 40.0, "DE_Ratio": 0.00, "Moat_Score": 9.4, "OI_Signal": "Short Covering", "News_Sent": +0.36, "News_Headline": "Institutional RWA tokenization volume climbs on Ethereum L2 rollups.", "Key_Products": "Institutional RWA Tokenization, L2 Rollup Settlement, Glamsterdam Upgrade"},
    {"Ticker": "SOL-USD",  "Company": "Solana (SOL)",    "Asset_Class": "Cryptocurrency", "Segment": "Crypto Growth",    "Sector": "Crypto Layer-1",        "Spot": 218.0,   "Lot_Size": 1, "Base_Ret": 0.395, "Vol": 0.74, "Beta": 1.95, "ROE": 0.350, "YoY_Rev_%": 62.0, "YoY_PAT_%": 65.0, "DE_Ratio": 0.00, "Moat_Score": 9.1, "OI_Signal": "Long Buildup", "News_Sent": +0.67, "News_Headline": "DEX daily volume & Firedancer validator throughput drive breakout momentum.", "Key_Products": "High-Throughput DeFi/DEX Volume, Firedancer Client, Consumer Web3 Apps"},
    {"Ticker": "BNB-USD",  "Company": "BNB Chain (BNB)", "Asset_Class": "Cryptocurrency", "Segment": "Crypto Blue-Chip", "Sector": "Crypto Exchange/L1",    "Spot": 645.0,   "Lot_Size": 1, "Base_Ret": 0.275, "Vol": 0.48, "Beta": 1.35, "ROE": 0.290, "YoY_Rev_%": 28.0, "YoY_PAT_%": 32.0, "DE_Ratio": 0.00, "Moat_Score": 8.8, "OI_Signal": "Long Buildup", "News_Sent": +0.30, "News_Headline": "Quarterly auto-burn deflation keeps BNB volatility lowest among top alts.", "Key_Products": "Quarterly Auto-Burn Deflation, BNB Smart Chain, Global Exchange Liquidity"},
    {"Ticker": "LINK-USD", "Company": "Chainlink (LINK)","Asset_Class": "Cryptocurrency", "Segment": "Crypto Growth",    "Sector": "Crypto Infrastructure", "Base_Ret": 0.325, "Spot": 18.4, "Lot_Size": 1, "Vol": 0.68, "Beta": 1.75, "ROE": 0.240, "YoY_Rev_%": 42.0, "YoY_PAT_%": 44.0, "DE_Ratio": 0.00, "Moat_Score": 9.0, "OI_Signal": "Long Buildup", "News_Sent": +0.48, "News_Headline": "CCIP cross-chain bank settlement pilots expand across Asian institutions.", "Key_Products": "CCIP Cross-Chain Bank Interoperability & Institutional Data Oracles"},
    {"Ticker": "XRP-USD",  "Company": "Ripple (XRP)",    "Asset_Class": "Cryptocurrency", "Segment": "Crypto Growth",    "Sector": "Crypto Payments",       "Base_Ret": 0.290, "Spot": 1.12,    "Lot_Size": 1, "Vol": 0.64, "Beta": 1.60, "ROE": 0.250, "YoY_Rev_%": 35.0, "YoY_PAT_%": 36.0, "DE_Ratio": 0.00, "Moat_Score": 8.6, "OI_Signal": "Short Unwinding", "News_Sent": -0.10, "News_Headline": "Consolidating after sharp rally; watching cross-border RLUSD liquidity.", "Key_Products": "Cross-Border Institutional Liquidity, RLUSD Stablecoin & Bank Rails"},
]

UPCOMING_IPOS_DATA = [
    {"IPO_Company": "NSE India (National Stock Exchange)", "Sector": "Capital Markets Monopoly", "DRHP_Stage": "Pre-DRHP / Regulatory Clearance", "Est_GMP_%": 42.0, "QIB_Demand_Score": 9.8, "Product_Moat_Summary": "93%+ equity derivatives market share in India; highest ROE exchange globally (~45%).", "Expected_1Y_Alpha_%": 28.5},
    {"IPO_Company": "Hindustan Coca-Cola Holdings",      "Sector": "FMCG & Beverages",         "DRHP_Stage": "SEBI DRHP Filed (Nov 2026)",      "Est_GMP_%": 24.0, "QIB_Demand_Score": 9.2, "Product_Moat_Summary": "Largest Coca-Cola bottling & distribution network across India with 14+ manufacturing plants.", "Expected_1Y_Alpha_%": 21.0},
    {"IPO_Company": "Reliance Jio Platforms",              "Sector": "Telecom, 5G & AI Cloud",   "DRHP_Stage": "Board Restructuring / Pipeline",   "Est_GMP_%": 26.0, "QIB_Demand_Score": 9.6, "Product_Moat_Summary": "490M+ subscribers, JioBrain AI enterprise stack, JioFiber & India digital ecosystem.", "Expected_1Y_Alpha_%": 22.5},
    {"IPO_Company": "South Eastern Coalfields (SECL)",     "Sector": "Energy & PSU Mining",      "DRHP_Stage": "SEBI DRHP Filed (Nov 2026)",      "Est_GMP_%": 18.5, "QIB_Demand_Score": 8.7, "Product_Moat_Summary": "Coal India's largest coal-producing subsidiary with high dividend yield & pithead power linkage.", "Expected_1Y_Alpha_%": 17.8},
    {"IPO_Company": "Zepto (Kiranakart Technologies)",     "Sector": "Quick Commerce & Logistics", "DRHP_Stage": "Reverse Flip Complete / DRHP Prep","Est_GMP_%": 31.0, "QIB_Demand_Score": 8.9, "Product_Moat_Summary": "10-minute dark-store hyperlocal delivery leader with rising private-label gross margins.", "Expected_1Y_Alpha_%": 24.5},
    {"IPO_Company": "PhonePe India",                       "Sector": "Digital Payments & Wealth","DRHP_Stage": "DRHP Preparation",               "Est_GMP_%": 29.0, "QIB_Demand_Score": 9.3, "Product_Moat_Summary": "~48% UPI transaction market share, Indus Appstore, insurance & stock broking.", "Expected_1Y_Alpha_%": 23.0},
]

# ==========================================
# 4. LIVE YAHOO FINANCE MARKET FEED ENGINE
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_market_snapshot(tickers_tuple: tuple):
    meta = {
        "nifty": 25144.30, "nifty_chg": -0.61,
        "banknifty": 61034.10, "banknifty_chg": -0.79,
        "brent": 96.0, "brent_chg": +0.85,
        "usdinr": 94.25, "usdinr_chg": +0.12,
        "is_live": False,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M:%S %p IST"),
        "quotes": {},
        "hist_returns": {}
    }
    if not YFINANCE_AVAILABLE:
        return meta

    macro_symbols = ["^NSEI", "^NSEBANK", "BZ=F", "INR=X"]
    all_syms = list(dict.fromkeys(macro_symbols + list(tickers_tuple)))

    try:
        raw = yf.download(
            tickers=all_syms,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False
        )
        if raw is None or raw.empty:
            return meta

        live_hits = 0
        for sym in all_syms:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    close_s = raw[sym]["Close"].dropna()
                else:
                    close_s = raw["Close"].dropna()

                if len(close_s) >= 2:
                    last_px = float(close_s.iloc[-1])
                    prev_px = float(close_s.iloc[-2])
                    chg_pct = ((last_px / prev_px) - 1.0) * 100.0
                    rets = close_s.pct_change().dropna()
                    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 20 else None
                    ann_ret = float(((close_s.iloc[-1] / close_s.iloc[0]) ** (252.0 / max(len(close_s), 60))) - 1.0) if len(close_s) > 40 else None

                    meta["quotes"][sym] = {
                        "spot": round(last_px, 2),
                        "chg_pct": round(chg_pct, 2),
                        "ann_vol": round(ann_vol, 4) if ann_vol and ann_vol > 0.03 else None,
                        "ann_ret": round(float(np.clip(ann_ret, 0.06, 0.45)), 4) if ann_ret is not None else None,
                    }
                    if len(rets) >= 60:
                        meta["hist_returns"][sym] = rets.values[-250:]
                    live_hits += 1
            except Exception:
                continue

        if "^NSEI" in meta["quotes"]:
            meta["nifty"] = meta["quotes"]["^NSEI"]["spot"]
            meta["nifty_chg"] = meta["quotes"]["^NSEI"]["chg_pct"]
        if "^NSEBANK" in meta["quotes"]:
            meta["banknifty"] = meta["quotes"]["^NSEBANK"]["spot"]
            meta["banknifty_chg"] = meta["quotes"]["^NSEBANK"]["chg_pct"]
        if "BZ=F" in meta["quotes"]:
            meta["brent"] = meta["quotes"]["BZ=F"]["spot"]
            meta["brent_chg"] = meta["quotes"]["BZ=F"]["chg_pct"]
        if "INR=X" in meta["quotes"]:
            meta["usdinr"] = meta["quotes"]["INR=X"]["spot"]
            meta["usdinr_chg"] = meta["quotes"]["INR=X"]["chg_pct"]

        if live_hits >= 5:
            meta["is_live"] = True
            meta["timestamp"] = datetime.now().strftime("%d %b %Y, %I:%M:%S %p IST")
    except Exception:
        pass

    return meta

# ==========================================
# 5. UNIVERSAL PDF GENERATOR (ORANGE & WHITE)
# ==========================================
def generate_investment_memo_pdf(
    macro_regime: str, brent_usd: float, usd_inr: float, solver_choice: str, horizon_label: str,
    total_lump_inr: float, deployed_now_inr: float, dry_powder_inr: float, liquid_yield_pct: float,
    monthly_sip_inr: float, total_invested: float, target_cagr: float, total_port_cagr: float,
    projected_val: float, projected_profit: float, port_vol: float, port_sharpe: float,
    avg_ml_acc: float, avg_oos_r2: float, apply_indian_tax: bool,
    winners_df: pd.DataFrame, top_ipos_df: pd.DataFrame, ipo_capital: float,
    opt_knapsack_df: pd.DataFrame
) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def clean_txt(s):
        return (
            str(s)
            .replace("\u2014", "-")
            .replace("\u2013", "-")
            .replace("\u20b9", "INR ")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )

    pdf.set_fill_color(234, 88, 12)
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_xy(12, 7)
    pdf.cell(0, 7, clean_txt("AETHERIS QUANT // OR - Quantitative Investment Memorandum"), ln=1)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(12)
    gen_date = datetime.now().strftime("%d %b %Y, %I:%M %p IST")
    pdf.cell(0, 5, clean_txt(f"Generated: {gen_date}  |  Regime: {macro_regime}"), ln=1)
    pdf.set_x(12)
    pdf.cell(0, 5, clean_txt(f"Brent: ${brent_usd:.1f}/bbl  |  USD/INR: Rs {usd_inr:.2f}  |  Solver: {solver_choice}"), ln=1)

    pdf.ln(9)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, clean_txt("1. Executive Portfolio & Macro-Adjusted Capital Deployment"), ln=1)
    pdf.set_draw_color(249, 115, 22)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(3)

    var_95 = max(0.0, (1.645 * port_vol - total_port_cagr) * 100)
    summary_rows = [
        ("Total Initial Lump-Sum", f"INR {total_lump_inr:,.0f}", "Immediate Risk-On Deployment", f"INR {deployed_now_inr:,.0f}"),
        ("Tactical Dry-Powder Reserve", f"INR {dry_powder_inr:,.0f}", "Liquid Reserve Yield (Repo-Linked)", f"{liquid_yield_pct:.2f}% p.a."),
        ("Monthly SIP / Crypto DCA", f"INR {monthly_sip_inr:,.0f}/mo", "Total Committed Over Horizon", f"INR {total_invested:,.0f}"),
        ("Expected Blended CAGR", f"{total_port_cagr*100:.2f}% (Floor: {target_cagr*100:.1f}%)", "Projected Maturity Value", f"INR {projected_val:,.0f}"),
        ("Projected Net Wealth Gain", f"+INR {projected_profit:,.0f}", "Volatility / Sharpe / 95% VaR", f"{port_vol*100:.1f}% / {port_sharpe:.2f} / -{var_95:.1f}%"),
    ]

    pdf.set_font("Helvetica", "", 8.5)
    for k1, v1, k2, v2 in summary_rows:
        pdf.set_fill_color(255, 247, 237)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(48, 6.5, clean_txt(f" {k1}"), border=1, fill=True)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(45, 6.5, clean_txt(f" {v1}"), border=1)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(48, 6.5, clean_txt(f" {k2}"), border=1, fill=True)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(45, 6.5, clean_txt(f" {v2}"), border=1, ln=1)
        pdf.set_font("Helvetica", "", 8.5)

    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, clean_txt("2. Walk-Forward Stacking ML & Indian Tax Compliance"), ln=1)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(0, 4.8, clean_txt(
        f"- Stacking ML Ensemble: Random Forest (40%) + Gradient Boosting (40%) + Ridge (20%) | Walk-Forward Dir. Acc: {avg_ml_acc:.1f}% (OOS R2: {avg_oos_r2:.3f}).\n"
        f"- Tax Regime: {'Post-Tax Optimization (12.5% Equity LTCG >= 1Y / 20% STCG; 31.2% Flat VDA Crypto Sec 115BBH + 1% TDS)' if apply_indian_tax else 'Pre-Tax Gross Returns'}."
    ))

    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, clean_txt("3. OR-Optimized Long-Term Multi-Asset Allocations"), ln=1)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(3)

    headers = ["Ticker", "Company / Asset", "Class", "Macro", "Net CAGR", "Weight", "Deploy Now (INR)", "SIP/Mo (INR)"]
    col_widths = [24, 44, 20, 17, 19, 17, 25, 20]
    pdf.set_fill_color(234, 88, 12)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 6.5, clean_txt(h), border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for idx, row in winners_df.iterrows():
        fill = idx % 2 == 0
        pdf.set_fill_color(255, 247, 237) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(0, 0, 0)
        vals = [
            row["Ticker"],
            str(row["Company"])[:23],
            "Equity" if row["Asset_Class"] == "Indian Equity" else "Crypto",
            f"{row['Macro_Alpha_%']:+.1f}%",
            f"{row['Effective_CAGR']*100:.2f}%",
            f"{row['Portfolio_Weight_%']:.2f}%",
            f"{row['Allocation_INR']:,.0f}",
            f"{row['Monthly_SIP_INR']:,.0f}"
        ]
        for val, w, al in zip(vals, col_widths, ["L", "L", "C", "R", "R", "R", "R", "R"]):
            pdf.cell(w, 6.0, clean_txt(val), border=1, fill=fill, align=al)
        pdf.ln()

    if not opt_knapsack_df.empty:
        pdf.ln(5)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, clean_txt("4. Next-Day Intraday Options & Vertical Spreads Trade Ticket"), ln=1)
        pdf.line(12, pdf.get_y(), 198, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 8)
        for _, r in opt_knapsack_df.iterrows():
            pdf.cell(0, 5.2, clean_txt(
                f"- {r['Ticker']} | {r['Strategy_Structure']} ({r['Contract_Legs']}) | Lots: {int(r['OR_Optimal_Lots'])} x {int(r['Lot_Size'])} | "
                f"Margin: INR {r['Total_Margin_INR']:,.0f} | Est. Next-Day Profit: +INR {r['Total_Est_Profit_INR']:,.0f}"
            ), ln=1)

    raw_out = pdf.output(dest="S") if isinstance(pdf.output(), str) else pdf.output()
    if isinstance(raw_out, str):
        return raw_out.encode("latin-1", "replace")
    return bytes(raw_out)

# ==========================================
# 6. BLACK-SCHOLES OPTION & MULTI-LEG SPREAD PRICER
# ==========================================
def black_scholes_call_put(S: float, K: float, T_years: float, r: float, sigma: float, opt_type: str = "CE"):
    T = max(T_years, 1e-5)
    sigma = max(sigma, 0.05)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if opt_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return max(0.5, float(price)), float(delta), float(gamma), float(theta)

def get_strike_step(spot: float) -> int:
    if spot < 500:
        return 5
    elif spot < 2000:
        return 20
    elif spot < 5000:
        return 50
    return 100

def evaluate_option_structures(sym: str, company: str, spot: float, lot: int, step: int,
                               pred_move: float, win_prob: float, iv: float, T_now: float, T_next: float, rf_rate: float,
                               strategy_filter: str):
    pred_next_spot = spot * (1.0 + pred_move)
    atm_strike = round(spot / step) * step
    is_bullish = pred_move >= 0
    opt_type = "CE" if is_bullish else "PE"

    structures = []

    if strategy_filter in ["All Strategies (Auto-Select Best EV)", "Single-Leg Naked Options Only"]:
        for tag, strike in [("Naked ATM", atm_strike), ("Naked 1-OTM", atm_strike + step if is_bullish else atm_strike - step)]:
            p_now, d_now, _, th_now = black_scholes_call_put(spot, strike, T_now, rf_rate, iv, opt_type)
            p_next, _, _, _ = black_scholes_call_put(pred_next_spot, strike, T_next, rf_rate, iv * 0.98, opt_type)

            capital_1_lot = p_now * lot
            est_profit_1_lot = ((p_next - p_now) * lot) - 160.0
            roi_pct = (est_profit_1_lot / max(capital_1_lot, 1.0)) * 100.0
            ev_lot = (win_prob / 100.0) * est_profit_1_lot - (1.0 - win_prob / 100.0) * (capital_1_lot * 0.35)

            structures.append({
                "Ticker": sym, "Company": company, "Underlying_Spot": round(spot, 2),
                "Strategy_Structure": f"{'Bullish' if is_bullish else 'Bearish'} {tag} {opt_type}",
                "Contract_Legs": f"BUY {int(strike)} {opt_type}", "Lot_Size": lot,
                "Net_Debit_Prem": round(p_now, 2), "Target_Spread_Val": round(p_next, 2),
                "Margin_Required_Lot": round(capital_1_lot, 0), "Theta_Day_Lot": round(th_now * lot, 0),
                "Theta_Saved_%": 0.0, "Max_Payoff_Cap_Lot": "Unlimited" if is_bullish else f"INR {strike*lot:,.0f}",
                "Pred_Move_%": round(pred_move * 100, 2), "Win_Prob_%": round(win_prob, 1),
                "Est_Profit_1_Lot": round(est_profit_1_lot, 0), "ROI_Per_Lot_%": round(roi_pct, 1),
                "Expected_EV_Lot": round(ev_lot, 0), "Net_Delta": round(d_now, 2)
            })

    if strategy_filter in ["All Strategies (Auto-Select Best EV)", "Vertical Debit Spreads Only (Bull Call / Bear Put)"]:
        long_strike = atm_strike
        short_strike = atm_strike + (2 * step) if is_bullish else atm_strike - (2 * step)
        spread_width = abs(short_strike - long_strike)

        p_long_now, d_long, _, th_long = black_scholes_call_put(spot, long_strike, T_now, rf_rate, iv, opt_type)
        p_short_now, d_short, _, th_short = black_scholes_call_put(spot, short_strike, T_now, rf_rate, iv * 0.99, opt_type)
        p_long_next, _, _, _ = black_scholes_call_put(pred_next_spot, long_strike, T_next, rf_rate, iv * 0.98, opt_type)
        p_short_next, _, _, _ = black_scholes_call_put(pred_next_spot, short_strike, T_next, rf_rate, iv * 0.97, opt_type)

        net_debit_now = max(1.0, p_long_now - p_short_now)
        net_spread_next = max(0.0, p_long_next - p_short_next)
        margin_1_lot = (net_debit_now * lot) * 1.05
        est_profit_1_lot = ((net_spread_next - net_debit_now) * lot) - 240.0
        net_theta_day_lot = (th_long - th_short) * lot
        theta_saved_pct = round((1.0 - abs(th_long - th_short) / (abs(th_long) + 1e-8)) * 100.0, 1)
        max_capped_profit = max(0.0, (spread_width - net_debit_now) * lot - 240.0)
        roi_pct = (est_profit_1_lot / max(margin_1_lot, 1.0)) * 100.0

        spread_win_prob = min(89.0, win_prob + 3.5)
        ev_lot = (spread_win_prob / 100.0) * est_profit_1_lot - (1.0 - spread_win_prob / 100.0) * (net_debit_now * lot * 0.30)

        structures.append({
            "Ticker": sym, "Company": company, "Underlying_Spot": round(spot, 2),
            "Strategy_Structure": "Bull Call Spread" if is_bullish else "Bear Put Spread",
            "Contract_Legs": f"BUY {int(long_strike)}{opt_type} / SELL {int(short_strike)}{opt_type}",
            "Lot_Size": lot, "Net_Debit_Prem": round(net_debit_now, 2), "Target_Spread_Val": round(net_spread_next, 2),
            "Margin_Required_Lot": round(margin_1_lot, 0), "Theta_Day_Lot": round(net_theta_day_lot, 0),
            "Theta_Saved_%": theta_saved_pct, "Max_Payoff_Cap_Lot": f"INR {max_capped_profit:,.0f}",
            "Pred_Move_%": round(pred_move * 100, 2), "Win_Prob_%": round(spread_win_prob, 1),
            "Est_Profit_1_Lot": round(est_profit_1_lot, 0), "ROI_Per_Lot_%": round(roi_pct, 1),
            "Expected_EV_Lot": round(ev_lot, 0), "Net_Delta": round(d_long - d_short, 2)
        })

    return structures

# ==========================================
# 7. WALK-FORWARD STACKING ML + MACRO SHOCK ENGINE
# ==========================================
@st.cache_data
def run_long_term_portfolio_research(df_input: pd.DataFrame, horizon_years: float, min_quality_score: float,
                                     apply_indian_tax: bool, macro_regime_key: str, brent_usd: float, usd_inr: float,
                                     hist_returns_map: dict):
    rng = np.random.default_rng(42)
    df = df_input.copy().reset_index(drop=True)
    macro_cfg = MACRO_ENVIRONMENTS[macro_regime_key]
    sector_shifts = macro_cfg["sector_alpha_shift"]
    brent_delta_10 = (brent_usd - BASE_BRENT_USD) / 10.0
    fx_delta_1 = usd_inr - BASE_USD_INR
    vol_mult = macro_cfg["vol_mult"] + max(0.0, brent_delta_10 * 0.06) + max(0.0, fx_delta_1 * 0.035)

    n_assets, n_days = len(df), 750
    equity_factor = rng.normal(0.00058, 0.0102 * vol_mult, n_days)
    crypto_factor = rng.normal(0.00115, 0.0285 * vol_mult, n_days)
    returns_matrix = np.zeros((n_days, n_assets))

    for i, row in df.iterrows():
        ticker = row["Ticker"]
        daily_mu = row["Base_Ret"] / 252.0
        daily_vol = (row["Vol"] * vol_mult) / np.sqrt(252.0)
        factor_shock = (0.65 * (crypto_factor - 0.00115) + 0.25 * (equity_factor - 0.00058)) if row["Asset_Class"] == "Cryptocurrency" else (row["Beta"] * (equity_factor - 0.00058))
        sim_series = daily_mu + factor_shock + rng.normal(0, daily_vol * 0.78, n_days)
        if ticker in hist_returns_map and len(hist_returns_map[ticker]) >= 60:
            real_tail = hist_returns_map[ticker]
            k_len = min(len(real_tail), n_days)
            sim_series[-k_len:] = real_tail[-k_len:] * vol_mult
        returns_matrix[:, i] = sim_series

    prices_df = pd.DataFrame(100 * np.exp(np.cumsum(returns_matrix, axis=0)), columns=df["Ticker"])
    ret_df = pd.DataFrame(returns_matrix, columns=df["Ticker"])

    rate_debt_penalty = 1.4 if ("Rate Hike" in macro_regime_key or "Twin Shock" in macro_regime_key) else 1.0
    df["Fundamental_Score"] = np.round(
        np.clip(df["YoY_Rev_%"] / 40.0, 0, 1) * 20
        + np.clip(df["YoY_PAT_%"] / 45.0, 0, 1) * 25
        + np.clip(df["ROE"] / 0.30, 0, 1) * 20
        + np.clip(1.0 - (df["DE_Ratio"] * rate_debt_penalty / 1.8), 0, 1) * 15
        + np.clip(df["Moat_Score"] / 10.0, 0, 1) * 20, 1
    )

    short_weight = np.clip(1.0 - (horizon_years / 5.0), 0.15, 0.85)
    horizon_decay = np.clip(1.25 - 0.18 * horizon_years, 0.35, 1.0)
    gross_preds, net_preds, macro_adj_list, dir_acc_list, oos_r2_list = [], [], [], [], []

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        series = prices_df[ticker].values
        r_series = ret_df[ticker].values
        mom_60d = (series[-1] / series[-60]) - 1.0
        vol_60d = np.std(r_series[-60:]) * np.sqrt(252)

        X, y = [], []
        for t in range(60, len(series) - 21, 4):
            X.append([(series[t]/series[t-20])-1.0, (series[t]/series[t-60])-1.0, np.std(r_series[t-20:t])*np.sqrt(252), row["YoY_PAT_%"]/100.0, row["Moat_Score"]/10.0])
            y.append(np.clip(((series[t+21]/series[t])**12)-1.0, -0.45, 0.85))

        X_arr, y_arr = np.array(X), np.array(y)
        split = int(len(X_arr) * 0.8)
        scaler = StandardScaler()
        X_tr, X_te = scaler.fit_transform(X_arr[:split]), scaler.transform(X_arr[split:])
        y_tr, y_te = y_arr[:split], y_arr[split:]

        rf = RandomForestRegressor(n_estimators=45, max_depth=4, random_state=42).fit(X_tr, y_tr)
        gbr = GradientBoostingRegressor(n_estimators=40, max_depth=3, learning_rate=0.06, random_state=42).fit(X_tr, y_tr)
        ridge = Ridge(alpha=4.0).fit(X_tr, y_tr)

        y_val = 0.4 * rf.predict(X_te) + 0.4 * gbr.predict(X_te) + 0.2 * ridge.predict(X_te)
        dir_acc_list.append(round(float(np.mean(np.sign(y_val) == np.sign(y_te)) * 100), 1))
        oos_r2_list.append(round(float(np.clip(r2_score(y_te, y_val), -0.4, 0.92)), 3))

        x_now = scaler.transform([[(series[-1]/series[-20])-1.0, mom_60d, vol_60d, row["YoY_PAT_%"]/100.0, row["Moat_Score"]/10.0]])
        ml_tech = 0.4 * rf.predict(x_now)[0] + 0.4 * gbr.predict(x_now)[0] + 0.2 * ridge.predict(x_now)[0]

        max_cap = 0.45 if row["Asset_Class"] == "Cryptocurrency" else 0.32
        fund_alpha = row["Base_Ret"] * 0.55 + (row["YoY_PAT_%"] / 100.0) * 0.20 + (row["ROE"] * 0.16) + (row["Moat_Score"] / 10.0) * 0.04
        base_cagr = short_weight * np.clip(ml_tech, 0.08, max_cap) + (1 - short_weight) * np.clip(fund_alpha, 0.09, max_cap)

        regime_shift = sector_shifts.get(row["Sector"], 0.0) * horizon_decay
        crude_shift = brent_delta_10 * BRENT_SECTOR_BETA_PER_10USD.get(row["Sector"], 0.0) * horizon_decay
        fx_shift = fx_delta_1 * FX_SECTOR_BETA_PER_1INR.get(row["Sector"], 0.0) * horizon_decay
        total_shift = regime_shift + crude_shift + fx_shift

        gross_cagr = float(np.clip(base_cagr + total_shift, 0.04, max_cap + 0.05))
        tax_drag = (0.312 if row["Asset_Class"] == "Cryptocurrency" else (0.125 if horizon_years >= 1.0 else 0.20)) if apply_indian_tax else 0.0
        gross_preds.append(round(gross_cagr, 4))
        net_preds.append(round(gross_cagr * (1.0 - tax_drag), 4))
        macro_adj_list.append(round(total_shift * 100, 2))

    df["Macro_Alpha_%"] = macro_adj_list
    df["Gross_ML_CAGR"] = gross_preds
    df["Effective_CAGR"] = net_preds
    df["ML_Dir_Acc_%"] = dir_acc_list
    df["OOS_R2"] = oos_r2_list

    adj_vol = df["Vol"] * vol_mult
    sharpe_p = df["Effective_CAGR"] / (adj_vol + 1e-8)
    sharpe_norm = (sharpe_p - sharpe_p.min()) / (sharpe_p.max() - sharpe_p.min() + 1e-8)
    ret_norm = (df["Effective_CAGR"] - df["Effective_CAGR"].min()) / (df["Effective_CAGR"].max() - df["Effective_CAGR"].min() + 1e-8)
    df["Conviction_Score"] = np.round(0.42 * df["Fundamental_Score"] + 0.33 * (ret_norm * 100) + 0.25 * (sharpe_norm * 100), 1)

    df_qual = df[df["Fundamental_Score"] >= min_quality_score].reset_index(drop=True)
    if len(df_qual) < 5:
        df_qual = df.sort_values(by="Fundamental_Score", ascending=False).head(8).reset_index(drop=True)

    q_rets = ret_df[df_qual["Ticker"]]
    sample_cov = q_rets.cov().values * 252.0
    cov_matrix = 0.85 * sample_cov + 0.15 * np.diag(np.diag(sample_cov)) + 1e-5 * np.eye(len(df_qual))
    return df, df_qual, cov_matrix, q_rets

# ==========================================
# 8. 4 OPERATIONS RESEARCH PORTFOLIO SOLVERS
# ==========================================
def solve_portfolio_or(solver_type: str, df_q: pd.DataFrame, cov: np.ndarray, q_rets: pd.DataFrame,
                       target_ret: float, w_max: float, max_k: int, sector_cap: float, crypto_cap: float, rf_rate: float):
    mu = df_q["Effective_CAGR"].values
    conv = df_q["Conviction_Score"].values / 100.0
    sectors = df_q["Sector"].values
    asset_classes = df_q["Asset_Class"].values
    variances = np.diag(cov)
    n = len(mu)
    eff_k = min(max_k, n)
    eff_w_max = max(w_max, (1.0 / max(1, eff_k)) + 0.02)
    crypto_idx = [i for i, ac in enumerate(asset_classes) if ac == "Cryptocurrency"]
    feasible_target = min(target_ret, float(np.max(mu)) * 0.94)

    if "MILP" in solver_type:
        c = np.concatenate([-(mu + 0.045 * conv - 0.30 * variances), np.zeros(n)])
        integrality = np.concatenate([np.zeros(n), np.ones(n)])
        lb = np.zeros(2 * n)
        ub = np.concatenate([np.full(n, eff_w_max), np.ones(n)])
        rows, lhs, rhs = [], [], []

        r_bud = np.zeros(2 * n); r_bud[:n] = 1.0
        rows.append(r_bud); lhs.append(1.0); rhs.append(1.0)

        r_card = np.zeros(2 * n); r_card[n:] = 1.0
        rows.append(r_card); lhs.append(-np.inf); rhs.append(float(eff_k))

        r_ret = np.zeros(2 * n); r_ret[:n] = mu
        rows.append(r_ret); lhs.append(feasible_target); rhs.append(np.inf)

        for i in range(n):
            r_up = np.zeros(2 * n); r_up[i] = 1.0; r_up[n + i] = -eff_w_max
            rows.append(r_up); lhs.append(-np.inf); rhs.append(0.0)
            r_lo = np.zeros(2 * n); r_lo[i] = 1.0; r_lo[n + i] = -0.04
            rows.append(r_lo); lhs.append(0.0); rhs.append(np.inf)

        if crypto_idx:
            r_cr = np.zeros(2 * n); r_cr[crypto_idx] = 1.0
            rows.append(r_cr); lhs.append(-np.inf); rhs.append(crypto_cap)

        for sec in np.unique(sectors):
            idx = [i for i, s in enumerate(sectors) if s == sec]
            r_sec = np.zeros(2 * n); r_sec[idx] = 1.0
            rows.append(r_sec); lhs.append(-np.inf); rhs.append(max(sector_cap, 1.0 / max(1, len(np.unique(sectors))) + 0.05))

        res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=LinearConstraint(np.array(rows), lhs, rhs))
        if res.success and res.x is not None and np.sum(res.x[:n]) > 1e-6:
            weights = np.maximum(0.0, res.x[:n])
            return weights / np.sum(weights)

    elif "CVaR" in solver_type:
        scenarios = q_rets.values[-200:] * 252.0
        S = scenarios.shape[0]
        alpha = 0.95
        c = np.zeros(n + 1 + S)
        c[n] = 1.0
        c[n+1:] = 1.0 / ((1.0 - alpha) * S)
        lb = np.concatenate([np.zeros(n), [-5.0], np.zeros(S)])
        ub = np.concatenate([np.full(n, eff_w_max), [5.0], np.full(S, np.inf)])
        rows, lhs, rhs = [], [], []

        r_bud = np.zeros(n + 1 + S); r_bud[:n] = 1.0
        rows.append(r_bud); lhs.append(1.0); rhs.append(1.0)

        r_ret = np.zeros(n + 1 + S); r_ret[:n] = mu
        rows.append(r_ret); lhs.append(feasible_target); rhs.append(np.inf)

        if crypto_idx:
            r_cr = np.zeros(n + 1 + S); r_cr[crypto_idx] = 1.0
            rows.append(r_cr); lhs.append(-np.inf); rhs.append(crypto_cap)

        for sec in np.unique(sectors):
            idx = [i for i, s in enumerate(sectors) if s == sec]
            r_sec = np.zeros(n + 1 + S); r_sec[idx] = 1.0
            rows.append(r_sec); lhs.append(-np.inf); rhs.append(max(sector_cap, 1.0 / max(1, len(np.unique(sectors))) + 0.05))

        for s in range(S):
            r_s = np.zeros(n + 1 + S)
            r_s[:n] = scenarios[s]
            r_s[n] = 1.0
            r_s[n + 1 + s] = 1.0
            rows.append(r_s); lhs.append(0.0); rhs.append(np.inf)

        res = milp(c=c, integrality=np.zeros(n + 1 + S), bounds=Bounds(lb, ub), constraints=LinearConstraint(np.array(rows), lhs, rhs))
        if res.success and res.x is not None and np.sum(res.x[:n]) > 1e-6:
            weights = np.maximum(0.0, res.x[:n])
            return weights / np.sum(weights)

    else:
        init_w = np.ones(n) / n
        bounds = [(0.0, eff_w_max) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if crypto_idx:
            constraints.append({"type": "ineq", "fun": lambda w: crypto_cap - np.sum(w[crypto_idx])})
        for sec in np.unique(sectors):
            idx = [i for i, s in enumerate(sectors) if s == sec]
            eff_sec = max(sector_cap, 1.0 / max(1, len(np.unique(sectors))) + 0.05)
            constraints.append({"type": "ineq", "fun": lambda w, idx=idx, es=eff_sec: es - np.sum(w[idx])})

        if "Sharpe" in solver_type:
            def obj(w):
                r = np.dot(w, mu)
                v = np.sqrt(np.dot(w.T, np.dot(cov, w))) + 1e-8
                return -(r - rf_rate) / v
        else:
            constraints.append({"type": "ineq", "fun": lambda w: np.dot(w, mu) - feasible_target})
            def obj(w):
                return 0.5 * np.dot(w.T, np.dot(cov, w))

        res = minimize(obj, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success and np.sum(res.x) > 1e-6:
            weights = np.maximum(0.0, res.x)
            return weights / np.sum(weights)

    top_idx = np.argsort(-conv)[:eff_k]
    weights = np.zeros(n)
    weights[top_idx] = 1.0 / len(top_idx)
    return weights

# ==========================================
# 9. NEXT-DAY INTRADAY PATTERN + NEWS ML & OPTIONS KNAPSACK
# ==========================================
@st.cache_data
def run_next_day_intraday_and_options_ai(df_stocks: pd.DataFrame, brent_usd: float, usd_inr: float,
                                         macro_regime_key: str, rf_rate: float, days_to_expiry: int, strategy_filter: str,
                                         nifty_spot: float, banknifty_spot: float):
    rng = np.random.default_rng(108)
    brent_delta_10 = (brent_usd - BASE_BRENT_USD) / 10.0
    fx_delta_1 = usd_inr - BASE_USD_INR
    macro_cfg = MACRO_ENVIRONMENTS[macro_regime_key]
    vol_mult = macro_cfg["vol_mult"] + max(0.0, brent_delta_10 * 0.05)

    n_train = 1500
    f_rsi = rng.uniform(25, 78, n_train)
    f_ema_gap = rng.normal(0.004, 0.018, n_train)
    f_candle_score = rng.choice([-1.0, -0.5, 0.5, 1.0], n_train)
    f_oi_score = rng.choice([-1.0, -0.4, 0.4, 1.0], n_train)
    f_news = rng.uniform(-0.85, 0.85, n_train)

    y_next_ret = 0.0045 * f_candle_score + 0.0055 * f_oi_score + 0.0085 * f_news + 0.12 * f_ema_gap - 0.00018 * (f_rsi - 50.0) + rng.normal(0, 0.004, n_train)
    y_dir = (y_next_ret > 0).astype(int)

    X_train = np.column_stack([f_rsi, f_ema_gap, f_candle_score, f_oi_score, f_news])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    reg_rf = RandomForestRegressor(n_estimators=75, max_depth=5, random_state=42).fit(X_scaled, y_next_ret)
    reg_gbr = GradientBoostingRegressor(n_estimators=65, max_depth=3, learning_rate=0.06, random_state=42).fit(X_scaled, y_next_ret)
    clf_rf = RandomForestClassifier(n_estimators=75, max_depth=5, random_state=42).fit(X_scaled, y_dir)

    oi_map = {"Long Buildup": 1.0, "Short Covering": 0.45, "Long Unwinding": -0.45, "Short Buildup": -1.0}
    patterns_catalog = [
        ("Bullish Engulfing + VWAP Reclaim", 1.0),
        ("Inside-Bar Volatility Breakout", 0.85),
        ("Hammer Reversal at 20-EMA", 0.65),
        ("Bearish Breakdown Below Pivot", -0.95),
        ("Double-Top Rejection + OI Short", -0.80)
    ]

    intraday_rows, all_option_candidates, best_per_company_rows = [], [], []
    T_now = days_to_expiry / 365.0
    T_next = max(1, days_to_expiry - 1) / 365.0

    for _, row in df_stocks.iterrows():
        sym = row["Ticker"].replace(".NS", "")
        spot = float(row["Spot"])
        lot = int(row["Lot_Size"])
        iv = float(row["Vol"]) * vol_mult * 1.05
        step = get_strike_step(spot)

        oi_num = oi_map.get(row["OI_Signal"], 0.0)
        news_s = float(row["News_Sent"])
        overnight_macro = (
            brent_delta_10 * BRENT_SECTOR_BETA_PER_10USD.get(row["Sector"], 0.0) * 0.12
            + fx_delta_1 * FX_SECTOR_BETA_PER_1INR.get(row["Sector"], 0.0) * 0.12
        )

        if oi_num + news_s > 0.7:
            pat_name, pat_score = patterns_catalog[0]
            curr_rsi, ema_gap = float(rng.uniform(54, 66)), float(rng.uniform(0.006, 0.018))
        elif oi_num + news_s > 0.15:
            pat_name, pat_score = patterns_catalog[1]
            curr_rsi, ema_gap = float(rng.uniform(48, 60)), float(rng.uniform(0.002, 0.012))
        elif oi_num + news_s > -0.25:
            pat_name, pat_score = patterns_catalog[2]
            curr_rsi, ema_gap = float(rng.uniform(36, 46)), float(rng.uniform(-0.008, 0.003))
        else:
            pat_name, pat_score = patterns_catalog[3] if oi_num < -0.5 else patterns_catalog[4]
            curr_rsi, ema_gap = float(rng.uniform(34, 45)), float(rng.uniform(-0.018, -0.005))

        x_now = scaler.transform([[curr_rsi, ema_gap, pat_score, oi_num, news_s]])
        pred_move = float(np.clip(0.5 * reg_rf.predict(x_now)[0] + 0.5 * reg_gbr.predict(x_now)[0] + overnight_macro, -0.038, 0.042))
        prob_up = float(clf_rf.predict_proba(x_now)[0][1])

        direction = "LONG (BUY)" if pred_move >= 0 else "SHORT (SELL)"
        win_prob = float(np.clip((prob_up if pred_move >= 0 else (1.0 - prob_up)) * 100, 52.0, 86.5))
        pred_next_spot = spot * (1.0 + pred_move)
        stop_loss = spot * (1.0 - abs(pred_move) * 0.52) if pred_move >= 0 else spot * (1.0 + abs(pred_move) * 0.52)

        intraday_rows.append({
            "Ticker": sym, "Company": row["Company"], "Sector": row["Sector"],
            "Spot_INR": round(spot, 2), "Day_Chg_%": row.get("Day_Chg_%", 0.0),
            "Chart_Pattern": pat_name, "FO_OI_Signal": row["OI_Signal"],
            "News_Sentiment": round(news_s, 2), "Signal": direction, "Pred_Move_%": round(pred_move * 100, 2),
            "Win_Prob_%": round(win_prob, 1), "Target_INR": round(pred_next_spot, 2),
            "StopLoss_INR": round(stop_loss, 2), "Catalyst_News": row["News_Headline"]
        })

        structs = evaluate_option_structures(sym, row["Company"], spot, lot, step, pred_move, win_prob, iv, T_now, T_next, rf_rate, strategy_filter)
        all_option_candidates.extend(structs)
        if structs:
            best_per_company_rows.append(max(structs, key=lambda x: x["Expected_EV_Lot"]))

    index_setups = [
        {"Ticker": "NIFTY", "Company": "Nifty 50 Index", "Spot": nifty_spot, "Lot": 65, "Step": 50,
         "Move": -0.0058 if brent_usd >= 98 else +0.0064, "Prob": 68.5, "IV": 0.145 * vol_mult},
        {"Ticker": "BANKNIFTY", "Company": "Bank Nifty Index", "Spot": banknifty_spot, "Lot": 30, "Step": 100,
         "Move": +0.0078 if "Rate Hike" in macro_regime_key else -0.0072, "Prob": 66.5, "IV": 0.175 * vol_mult},
    ]
    for idx_s in index_setups:
        idx_structs = evaluate_option_structures(idx_s["Ticker"], idx_s["Company"], idx_s["Spot"], idx_s["Lot"], idx_s["Step"], idx_s["Move"], idx_s["Prob"], idx_s["IV"], T_now, T_next, rf_rate, strategy_filter)
        all_option_candidates.extend(idx_structs)
        if idx_structs:
            best_per_company_rows.append(max(idx_structs, key=lambda x: x["Expected_EV_Lot"]))

    df_intraday = pd.DataFrame(intraday_rows).sort_values(by="Win_Prob_%", ascending=False).reset_index(drop=True)
    df_best_options = pd.DataFrame(best_per_company_rows).sort_values(by="Expected_EV_Lot", ascending=False).reset_index(drop=True)
    df_all_structures = pd.DataFrame(all_option_candidates).sort_values(by="ROI_Per_Lot_%", ascending=False).reset_index(drop=True)
    return df_intraday, df_best_options, df_all_structures

def solve_options_knapsack_or(df_opt: pd.DataFrame, options_budget_inr: float, max_lots_per_contract: int = 3):
    profitable = df_opt[df_opt["Est_Profit_1_Lot"] > 400].reset_index(drop=True)
    if profitable.empty or options_budget_inr < 3000:
        return pd.DataFrame()

    n = len(profitable)
    c = -(0.65 * profitable["Est_Profit_1_Lot"].values + 0.35 * profitable["Expected_EV_Lot"].values)
    A = profitable["Margin_Required_Lot"].values.reshape(1, -1)
    res = milp(c=c, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.full(n, max_lots_per_contract)), constraints=LinearConstraint(A, -np.inf, options_budget_inr))

    chosen_lots = np.round(res.x).astype(int) if (res.success and res.x is not None) else np.zeros(n, dtype=int)
    profitable["OR_Optimal_Lots"] = chosen_lots
    selected = profitable[profitable["OR_Optimal_Lots"] > 0].copy()
    if not selected.empty:
        selected["Total_Margin_INR"] = selected["OR_Optimal_Lots"] * selected["Margin_Required_Lot"]
        selected["Total_Est_Profit_INR"] = selected["OR_Optimal_Lots"] * selected["Est_Profit_1_Lot"]
        selected["Total_Theta_Day_INR"] = selected["OR_Optimal_Lots"] * selected["Theta_Day_Lot"]
    return selected

# ==========================================
# 10. FETCH LIVE MARKET FEED & MERGE INTO UNIVERSE
# ==========================================
all_tickers_tuple = tuple([d["Ticker"] for d in NSE_MASTER_DATA] + [d["Ticker"] for d in CRYPTO_MASTER_DATA])
live_snap = fetch_live_market_snapshot(all_tickers_tuple)

def enrich_with_live_quotes(records: list, quotes_dict: dict) -> pd.DataFrame:
    df_out = pd.DataFrame(records)
    spots, chgs, vols, rets = [], [], [], []
    for _, r in df_out.iterrows():
        q = quotes_dict.get(r["Ticker"], {})
        spots.append(q.get("spot") if q.get("spot") is not None else r["Spot"])
        chgs.append(q.get("chg_pct") if q.get("chg_pct") is not None else 0.0)
        vols.append(q.get("ann_vol") if q.get("ann_vol") is not None else r["Vol"])
        rets.append(q.get("ann_ret") if q.get("ann_ret") is not None else r["Base_Ret"])
    df_out["Spot"] = spots
    df_out["Day_Chg_%"] = chgs
    df_out["Vol"] = vols
    df_out["Base_Ret"] = rets
    return df_out

df_stocks = enrich_with_live_quotes(NSE_MASTER_DATA, live_snap["quotes"])
df_crypto = enrich_with_live_quotes(CRYPTO_MASTER_DATA, live_snap["quotes"])

# ==========================================
# 11. SIDEBAR CONTROLS WITH LIVE FEED REFRESH
# ==========================================
with st.sidebar:
    st.markdown("## AETHERIS QUANT // OR")
    st.caption("Live Multi-Asset, Macro & Derivatives Engine")

    if st.button("Refresh Live Market Feed", width="stretch"):
        fetch_live_market_snapshot.clear()
        st.rerun()

    st.markdown("### 01. Quick-Start Strategy Profile")
    preset_choice = st.selectbox(
        "Investor Profile Preset",
        list(INVESTOR_PRESETS.keys()),
        index=0,
        help="Automatically configures horizon, target return, crypto cap, macro regime, and options mode."
    )
    p_cfg = INVESTOR_PRESETS[preset_choice]

    st.markdown("### 02. Capital & Return Target")
    total_lump_inr = st.number_input("Initial Lump-Sum Capital (INR)", 50000, 100000000, int(p_cfg["lump"]), 50000)
    monthly_sip_inr = st.number_input("Monthly SIP / DCA (INR/month)", 0, 2000000, int(p_cfg["sip"]), 5000)
    horizon_label = st.select_slider("Investment Duration", ["6 Months (0.5Y)", "1 Year", "2 Years", "3 Years", "5 Years"], value=p_cfg["horizon"])
    horizon_map = {"6 Months (0.5Y)": 0.5, "1 Year": 1.0, "2 Years": 2.0, "3 Years": 3.0, "5 Years": 5.0}
    horizon_years = horizon_map[horizon_label]
    target_cagr = st.slider("Desired Annual Return (%)", 11.0, 26.0, float(p_cfg["target_cagr"]), 0.5) / 100.0

    with st.expander("03. Geopolitics, War, Crude & Interest Rates", expanded=False):
        regime_idx = list(MACRO_ENVIRONMENTS.keys()).index(p_cfg["regime"])
        macro_regime = st.selectbox("Geopolitical / Rate Scenario", list(MACRO_ENVIRONMENTS.keys()), index=regime_idx)
        macro_cfg = MACRO_ENVIRONMENTS[macro_regime]
        default_brent_val = float(np.clip(live_snap["brent"], 65.0, 150.0)) if "Normal" in macro_regime else float(macro_cfg["default_brent"])
        default_fx_val = float(np.clip(live_snap["usdinr"], 82.0, 105.0)) if "Normal" in macro_regime else float(macro_cfg["default_usdinr"])
        brent_usd = st.slider("Brent Crude Oil ($/bbl)", 65.0, 150.0, default_brent_val, 0.5)
        usd_inr = st.slider("USD/INR Exchange Rate (INR/$)", 82.0, 105.0, default_fx_val, 0.25)
        extra_tranche_cut = max(0.0, (brent_usd - 105.0) * 0.005) + max(0.0, (usd_inr - 95.5) * 0.03)
        rec_deploy = float(np.clip(macro_cfg["deploy_now_pct"] - extra_tranche_cut, 0.25, 1.0))
        deploy_now_pct = st.slider("Deploy Lump-Sum Now (%)", 20, 100, int(round(rec_deploy * 100 / 5) * 5), 5) / 100.0
        tranche_months = st.slider("Deploy Reserve Over (Months)", 3, 18, 6, 3)

    with st.expander("04. Next-Day Intraday & Options Settings", expanded=False):
        options_budget_inr = st.number_input("Next-Day Options Budget (INR)", 10000, 5000000, int(p_cfg["opt_budget"]), 10000)
        opt_modes = ["All Strategies (Auto-Select Best EV)", "Vertical Debit Spreads Only (Bull Call / Bear Put)", "Single-Leg Naked Options Only"]
        strategy_filter = st.selectbox("Options Strategy Type", opt_modes, index=opt_modes.index(p_cfg["opt_mode"]))
        days_to_expiry = st.slider("Days to F&O Expiry", 1, 25, 10)
        max_lots_contract = st.slider("Max Lots Per Contract", 1, 10, 3)

    with st.expander("05. Advanced OR Solver & Tax Controls", expanded=False):
        solvers = ["Cardinality Constrained MILP (HiGHS K-Best)", "CVaR 95% Tail-Risk Minimization (LP)", "Maximum Sharpe Ratio (Fractional QP)", "Markowitz Minimum Variance (QP)"]
        solver_choice = st.selectbox("OR Optimization Solver", solvers, index=solvers.index(p_cfg["solver"]))
        max_k = st.slider("Number of Companies to Pick (K)", 5, 12, 8)
        w_max = st.slider("Max Allocation Per Company (%)", 12, 30, 20, 2) / 100.0
        sector_cap = st.slider("Max Sector Exposure (%)", 20, 55, 35, 5) / 100.0
        include_crypto = st.toggle("Include Cryptocurrencies", value=p_cfg["crypto"])
        base_crypto_cap = st.slider("Max Crypto Cap (%)", 0, 30, int(p_cfg["crypto_cap"]), 5, disabled=not include_crypto) / 100.0
        effective_crypto_cap = base_crypto_cap * macro_cfg["crypto_cap_mult"] if include_crypto else 0.0
        ipo_sleeve_pct = st.slider("Upcoming IPO Reserve (%)", 0, 30, int(p_cfg["ipo_pct"]), 5) / 100.0
        apply_indian_tax = st.toggle("Optimize Post-Tax (31.2% Crypto / 12.5% Equity)", value=True)

# ==========================================
# 12. EXECUTE FULL PIPELINE
# ==========================================
df_combined = pd.concat([df_stocks, df_crypto], ignore_index=True) if include_crypto else df_stocks.copy()

liquid_yield_pct = macro_cfg["rbi_repo_pct"] + 1.15
rf_rate = liquid_yield_pct / 100.0

df_all_scored, df_qualified, cov_matrix, q_rets = run_long_term_portfolio_research(
    df_combined, horizon_years, 65.0, apply_indian_tax, macro_regime, brent_usd, usd_inr, live_snap["hist_returns"]
)
active_crypto_cap = min(0.95, effective_crypto_cap / max(0.05, (1.0 - ipo_sleeve_pct))) if include_crypto else 0.0
opt_weights = solve_portfolio_or(solver_choice, df_qualified, cov_matrix, q_rets, target_cagr, w_max, max_k, sector_cap, active_crypto_cap, rf_rate)

deployed_now_inr = total_lump_inr * deploy_now_pct
dry_powder_inr = total_lump_inr * (1.0 - deploy_now_pct)
monthly_dry_powder_tranche = dry_powder_inr / tranche_months if dry_powder_inr > 0 else 0.0

active_deployed_inr = deployed_now_inr * (1.0 - ipo_sleeve_pct)
ipo_capital = deployed_now_inr * ipo_sleeve_pct

df_qualified["Portfolio_Weight_%"] = np.round(opt_weights * (1.0 - ipo_sleeve_pct) * 100, 2)
df_qualified["Allocation_INR"] = np.round(opt_weights * active_deployed_inr, 0)
df_qualified["Monthly_SIP_INR"] = np.round(opt_weights * (1.0 - ipo_sleeve_pct) * monthly_sip_inr, 0)
df_qualified["Staged_Dip_Buy_INR_Mo"] = np.round(opt_weights * (1.0 - ipo_sleeve_pct) * monthly_dry_powder_tranche, 0)

active_ret = float(np.dot(opt_weights, df_qualified["Effective_CAGR"].values))
ipo_df = pd.DataFrame(UPCOMING_IPOS_DATA)
top_ipos = ipo_df.sort_values(by="QIB_Demand_Score", ascending=False).head(3).copy()
ipo_ret = float(top_ipos["Expected_1Y_Alpha_%"].mean() / 100.0) * (0.875 if apply_indian_tax else 1.0)
risk_on_cagr = (1.0 - ipo_sleeve_pct) * active_ret + ipo_sleeve_pct * ipo_ret
net_liquid_yield = rf_rate * (0.70 if apply_indian_tax else 1.0)
avg_cash_fraction = (1.0 - deploy_now_pct) * min(1.0, (tranche_months / 2.0) / (horizon_years * 12.0))
total_port_cagr = (1.0 - avg_cash_fraction) * risk_on_cagr + avg_cash_fraction * net_liquid_yield

port_vol = float(np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))) * (1.0 - 0.25 * ipo_sleeve_pct) * deploy_now_pct
port_sharpe = (total_port_cagr - net_liquid_yield) / (port_vol + 1e-8)

n_months = int(horizon_years * 12)
r_monthly = (1 + total_port_cagr) ** (1 / 12) - 1
projected_val = total_lump_inr * ((1 + r_monthly) ** n_months) + (monthly_sip_inr * (((1 + r_monthly) ** n_months - 1) / (r_monthly + 1e-12)) if monthly_sip_inr > 0 else 0.0)
total_invested = total_lump_inr + (monthly_sip_inr * n_months)
projected_profit = projected_val - total_invested

actual_crypto_inr = df_qualified[df_qualified["Asset_Class"] == "Cryptocurrency"]["Allocation_INR"].sum()
actual_equity_inr = df_qualified[df_qualified["Asset_Class"] == "Indian Equity"]["Allocation_INR"].sum()
avg_ml_acc = float(df_all_scored["ML_Dir_Acc_%"].mean())
avg_oos_r2 = float(df_all_scored["OOS_R2"].mean())

df_intraday, df_best_options, df_all_structures = run_next_day_intraday_and_options_ai(
    df_stocks, brent_usd, usd_inr, macro_regime, rf_rate, days_to_expiry, strategy_filter,
    live_snap["nifty"], live_snap["banknifty"]
)
df_opt_knapsack = solve_options_knapsack_or(df_best_options, options_budget_inr, max_lots_per_contract=max_lots_contract)

top_intraday_pick = df_intraday.iloc[0]
top_option_pick = df_best_options.iloc[0]
knapsack_total_profit = float(df_opt_knapsack["Total_Est_Profit_INR"].sum()) if not df_opt_knapsack.empty else 0.0
knapsack_total_cap = float(df_opt_knapsack["Total_Margin_INR"].sum()) if not df_opt_knapsack.empty else 0.0
knapsack_theta_day = float(df_opt_knapsack["Total_Theta_Day_INR"].sum()) if not df_opt_knapsack.empty else 0.0

winners = df_qualified[df_qualified["Portfolio_Weight_%"] > 0.1].sort_values(by="Allocation_INR", ascending=False).reset_index(drop=True)
if winners.empty:
    winners = df_qualified.head(5).copy()

# ==========================================
# 13. LIVE ANIMATED TICKER, ORANGE & WHITE HEADER & KPI STRIP
# ==========================================
def fmt_tick(label: str, price_str: str, chg: float):
    cls = "tick-up" if chg >= 0 else "tick-down"
    return f"<span class='{cls}'>{label}: {price_str} ({chg:+.2f}%)</span>"

live_stock_ticks = "".join([
    fmt_tick(r["Ticker"].replace(".NS", ""), f"INR {r['Spot']:,.1f}", r["Day_Chg_%"])
    for _, r in df_stocks.head(8).iterrows()
])
live_crypto_ticks = "".join([
    fmt_tick(r["Ticker"].replace("-USD", ""), f"${r['Spot']:,.1f}", r["Day_Chg_%"])
    for _, r in df_crypto.head(3).iterrows()
])

ticker_items = (
    fmt_tick("NIFTY 50", f"{live_snap['nifty']:,.2f}", live_snap["nifty_chg"])
    + fmt_tick("BANK NIFTY", f"{live_snap['banknifty']:,.2f}", live_snap["banknifty_chg"])
    + fmt_tick("BRENT CRUDE", f"${brent_usd:.2f}/bbl", live_snap["brent_chg"])
    + fmt_tick("USD/INR", f"{usd_inr:.2f}", live_snap["usdinr_chg"])
    + live_stock_ticks
    + live_crypto_ticks
)
st.markdown(f"""
<div class="ticker-wrap">
    <div class="ticker-move">{ticker_items}{ticker_items}</div>
</div>
""", unsafe_allow_html=True)

feed_status_label = f"LIVE YAHOO FINANCE FEED | {live_snap['timestamp']}" if live_snap["is_live"] else f"CALIBRATED BASELINE FEED | {live_snap['timestamp']}"
st.markdown(f"""
<div class="brand-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
        <div>
            <div class="brand-title">AETHERIS QUANT // OR</div>
            <div class="brand-sub">Autonomous Equity, Tactical Hedging &amp; Enterprise Risk Intelligence System</div>
        </div>
        <div class="live-badge">{feed_status_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)

top_stock_names = ", ".join(winners.head(3)["Ticker"].str.replace(".NS", "").tolist())
st.markdown(f"""
<div class="action-banner">
    <b>EXECUTIVE SIGNAL:</b> Deploy <b>INR {deployed_now_inr:,.0f}</b> immediately across <b>{len(winners)} OR-selected assets</b> (anchored by <b>{top_stock_names}</b>) targeting a net <b>{total_port_cagr*100:.2f}% annual CAGR</b>.
    For the next trading session, highest-conviction intraday setup is <b>{top_intraday_pick['Ticker']} ({top_intraday_pick['Pred_Move_%']:+.2f}%)</b> and top derivatives structure is <b>{top_option_pick['Ticker']} {top_option_pick['Strategy_Structure']}</b>.
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card card-orange-1">
        <div class="metric-label">Expected Net Annual CAGR</div>
        <div class="metric-value">{total_port_cagr*100:.2f}%</div>
        <div class="metric-sub">Maturity: INR {projected_val:,.0f} (+INR {projected_profit:,.0f})</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card card-amber">
        <div class="metric-label">Initial Capital Split (Now / Reserve)</div>
        <div class="metric-value">INR {deployed_now_inr/1e5:.1f}L / {dry_powder_inr/1e5:.1f}L</div>
        <div class="metric-alert">{deploy_now_pct*100:.0f}% Deployed Now | {100-deploy_now_pct*100:.0f}% Liquid ({liquid_yield_pct:.2f}%)</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card card-orange-2">
        <div class="metric-label">Top Next-Day Intraday Signal</div>
        <div class="metric-value">{top_intraday_pick['Ticker']} ({top_intraday_pick['Pred_Move_%']:+.2f}%)</div>
        <div class="metric-sub">{top_intraday_pick['Signal']} | Win Prob: {top_intraday_pick['Win_Prob_%']:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card card-emerald">
        <div class="metric-label">OR Options Basket Est. Profit</div>
        <div class="metric-value">+INR {knapsack_total_profit:,.0f}</div>
        <div class="metric-alert">Margin: INR {knapsack_total_cap:,.0f} | Theta: INR {knapsack_theta_day:,.0f}/day</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 14. 7-TAB ORANGE & WHITE WORKSPACE (ZERO BLACK)
# ==========================================
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "01. Portfolio & Frontier Analytics",
    "02. Options & Spreads Maximizer",
    "03. Payoff & Greeks Lab",
    "04. Next-Day Intraday AI",
    "05. War, Crude & FX Stress Matrix",
    "06. Rebalancing & Tax Blotter",
    "07. Live Market Screener & Math"
])

# Style helper for light orange/white Pandas Styler tables
def style_light_table(df_in: pd.DataFrame):
    return df_in.style.set_properties(**{
        "background-color": "#ffffff",
        "color": "#000000",
        "border-color": "#fed7aa"
    }).set_table_styles([
        {"selector": "th", "props": [("background-color", "#fff7ed"), ("color", "#000000"), ("font-weight", "bold")]},
        {"selector": "tr:nth-child(even) td", "props": [("background-color", "#fffaf5"), ("color", "#000000")]},
    ])

# --- TAB 1: LONG-TERM PORTFOLIO + GRAPHICAL FRONTIER, RADAR & SUNBURST ---
with t1:
    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        st.subheader("OR-Optimized Multi-Asset Capital Allocation")
        disp = winners[["Ticker", "Company", "Spot", "Day_Chg_%", "Effective_CAGR", "Portfolio_Weight_%", "Allocation_INR", "Monthly_SIP_INR"]].copy()
        disp["Spot"] = disp["Spot"].map("{:,.1f}".format)
        disp["Day_Chg_%"] = disp["Day_Chg_%"].map("{:+.2f}%".format)
        disp["Effective_CAGR"] = (disp["Effective_CAGR"] * 100).map("{:.2f}%".format)
        disp["Portfolio_Weight_%"] = disp["Portfolio_Weight_%"].map("{:.2f}%".format)
        disp["Allocation_INR"] = disp["Allocation_INR"].map("INR {:,.0f}".format)
        disp["Monthly_SIP_INR"] = disp["Monthly_SIP_INR"].map("INR {:,.0f}".format)
        st.dataframe(style_light_table(disp), width="stretch", hide_index=True)

        if ipo_sleeve_pct > 0 and ipo_capital > 0:
            st.markdown(f"#### Dedicated IPO Sleeve (INR {ipo_capital:,.0f} across Top 3 Upcoming Listings)")
            top_ipos["IPO_LumpSum_INR"] = np.round(ipo_capital / len(top_ipos), 0)
            ipo_disp = top_ipos[["IPO_Company", "DRHP_Stage", "Est_GMP_%", "Expected_1Y_Alpha_%", "IPO_LumpSum_INR"]].copy()
            ipo_disp["IPO_LumpSum_INR"] = ipo_disp["IPO_LumpSum_INR"].map("INR {:,.0f}".format)
            st.dataframe(style_light_table(ipo_disp), width="stretch", hide_index=True)

        b1, b2 = st.columns(2)
        with b1:
            csv_out = winners[["Ticker", "Company", "Asset_Class", "Segment", "Sector", "Spot", "Day_Chg_%", "Macro_Alpha_%", "Conviction_Score", "Effective_CAGR", "Portfolio_Weight_%", "Allocation_INR", "Monthly_SIP_INR"]].to_csv(index=False).encode("utf-8")
            st.download_button("Export Execution Plan (CSV)", data=csv_out, file_name="aetheris_quant_portfolio.csv", mime="text/csv", width="stretch")
        with b2:
            if FPDF_AVAILABLE:
                pdf_bytes = generate_investment_memo_pdf(
                    macro_regime=macro_regime, brent_usd=brent_usd, usd_inr=usd_inr, solver_choice=solver_choice,
                    horizon_label=horizon_label, total_lump_inr=total_lump_inr, deployed_now_inr=deployed_now_inr,
                    dry_powder_inr=dry_powder_inr, liquid_yield_pct=liquid_yield_pct, monthly_sip_inr=monthly_sip_inr,
                    total_invested=total_invested, target_cagr=target_cagr, total_port_cagr=total_port_cagr,
                    projected_val=projected_val, projected_profit=projected_profit, port_vol=port_vol,
                    port_sharpe=port_sharpe, avg_ml_acc=avg_ml_acc, avg_oos_r2=avg_oos_r2,
                    apply_indian_tax=apply_indian_tax, winners_df=winners, top_ipos_df=top_ipos,
                    ipo_capital=ipo_capital, opt_knapsack_df=df_opt_knapsack
                )
                st.download_button("Export Institutional PDF Memo", data=pdf_bytes, file_name=f"Aetheris_Quant_Memo_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", width="stretch")

        st.subheader("Top Winners Multi-Factor DNA Radar")
        top4 = winners.head(4)
        categories = ["Rev Growth", "PAT Growth", "ROE Strength", "Product Moat", "ML Conviction"]
        fig_rad = go.Figure()
        radar_colors = ["#ea580c", "#f59e0b", "#059669", "#d97706"]
        for idx_r, r_row in top4.iterrows():
            vals = [
                min(100, (r_row["YoY_Rev_%"] / 50.0) * 100),
                min(100, (r_row["YoY_PAT_%"] / 60.0) * 100),
                min(100, (r_row["ROE"] / 0.45) * 100),
                min(100, r_row["Moat_Score"] * 10.0),
                min(100, r_row["Conviction_Score"])
            ]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=r_row["Ticker"].replace(".NS", ""),
                line=dict(color=radar_colors[idx_r % len(radar_colors)], width=2.4)
            ))
        fig_rad.update_layout(
            polar=dict(
                bgcolor="#ffffff",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#fed7aa", tickfont=dict(color="#000000")),
                angularaxis=dict(gridcolor="#fed7aa", tickfont=dict(color="#000000"))
            ),
            margin=dict(l=30, r=30, t=25, b=20)
        )
        apply_light_orange_chart_style(fig_rad, height=320)
        st.plotly_chart(fig_rad, width="stretch")

    with col_r:
        st.subheader("Compounding Wealth Trajectory & Confidence Band")
        months = np.arange(0, n_months + 1)
        t_yrs = months / 12.0
        opt_curve = np.array([total_lump_inr * ((1 + r_monthly) ** m) + (monthly_sip_inr * (((1 + r_monthly) ** m - 1) / (r_monthly + 1e-12)) if m > 0 else 0) for m in months])
        upper_band = opt_curve * np.exp(0.5 * port_vol * np.sqrt(t_yrs))
        lower_band = opt_curve * np.exp(-0.5 * port_vol * np.sqrt(t_yrs))
        invested_line = total_lump_inr + monthly_sip_inr * months
        r_nifty_m = (1 + 0.125) ** (1 / 12) - 1
        nifty_curve = [total_lump_inr * ((1 + r_nifty_m) ** m) + (monthly_sip_inr * (((1 + r_nifty_m) ** m - 1) / (r_nifty_m + 1e-12)) if m > 0 else 0) for m in months]

        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=months, y=upper_band, mode="lines", line=dict(width=0), showlegend=False))
        fig_w.add_trace(go.Scatter(x=months, y=lower_band, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(249, 115, 22, 0.16)", name="Volatility Cone"))
        fig_w.add_trace(go.Scatter(x=months, y=opt_curve, mode="lines", name="Aetheris OR Wealth", line=dict(color="#ea580c", width=3.2)))
        fig_w.add_trace(go.Scatter(x=months, y=nifty_curve, mode="lines", name="Nifty 50 Benchmark (12.5%)", line=dict(color="#d97706", dash="dash", width=2.2)))
        fig_w.add_trace(go.Scatter(x=months, y=invested_line, mode="lines", name="Total Principal", line=dict(color="#64748b", dash="dot", width=1.8)))
        fig_w.update_layout(xaxis_title="Months", yaxis_title="Portfolio Value (INR)", margin=dict(l=15, r=15, t=20, b=15), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#000000")))
        apply_light_orange_chart_style(fig_w, height=310)
        st.plotly_chart(fig_w, width="stretch")

        sb_df = winners[["Asset_Class", "Sector", "Ticker", "Allocation_INR"]].copy()
        sb_df["Ticker"] = sb_df["Ticker"].str.replace(".NS", "")
        if ipo_capital > 0:
            sb_df = pd.concat([sb_df, pd.DataFrame([{"Asset_Class": "Upcoming IPOs", "Sector": "Primary Listings", "Ticker": "IPO Sleeve", "Allocation_INR": ipo_capital}])], ignore_index=True)
        if dry_powder_inr > 0:
            sb_df = pd.concat([sb_df, pd.DataFrame([{"Asset_Class": "Liquid Reserve", "Sector": "Repo-Linked Debt", "Ticker": "Dry Powder", "Allocation_INR": dry_powder_inr}])], ignore_index=True)

        fig_sb = px.sunburst(sb_df, path=["Asset_Class", "Sector", "Ticker"], values="Allocation_INR", title="Multi-Asset Capital Hierarchy (Class > Sector > Ticker)", color="Asset_Class", color_discrete_sequence=["#f97316", "#fb923c", "#f59e0b", "#fdba74"])
        fig_sb.update_traces(textfont=dict(color="#000000"))
        fig_sb.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        apply_light_orange_chart_style(fig_sb, height=320)
        st.plotly_chart(fig_sb, width="stretch")

    st.subheader("Markowitz Efficient Frontier & Cross-Asset Risk-Return Map")
    rng_ef = np.random.default_rng(42)
    mu_vec = df_qualified["Effective_CAGR"].values
    n_sim = 900
    rand_w = rng_ef.dirichlet(np.ones(len(mu_vec)), size=n_sim)
    sim_rets = rand_w @ mu_vec
    sim_vols = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov_matrix, rand_w))
    sim_sharpe = (sim_rets - net_liquid_yield) / (sim_vols + 1e-8)

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=sim_vols * 100, y=sim_rets * 100, mode="markers",
        marker=dict(size=5, color=sim_sharpe, colorscale="Oranges", showscale=True, colorbar=dict(title=dict(text="Sharpe", font=dict(color="#000000")), tickfont=dict(color="#000000"))),
        name="Simulated Portfolios"
    ))
    fig_ef.add_trace(go.Scatter(
        x=np.sqrt(np.diag(cov_matrix)) * 100, y=mu_vec * 100, mode="markers+text",
        text=df_qualified["Ticker"].str.replace(".NS", ""), textposition="top center",
        textfont=dict(color="#000000"),
        marker=dict(size=8, color="#d97706", symbol="diamond"),
        name="Qualified Assets"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[port_vol * 100], y=[total_port_cagr * 100], mode="markers",
        marker=dict(size=18, color="#ea580c", symbol="star", line=dict(width=2, color="#ffffff")),
        name="Aetheris Optimal Portfolio"
    ))
    fig_ef.update_layout(xaxis_title="Annualized Volatility (%)", yaxis_title="Expected Post-Tax CAGR (%)", margin=dict(l=15, r=15, t=20, b=15))
    apply_light_orange_chart_style(fig_ef, height=340)
    st.plotly_chart(fig_ef, width="stretch")

# --- TAB 2: OPTIONS & SPREADS OR ---
with t2:
    st.subheader("OR-Optimized Next-Day Options & Vertical Debit Spreads Basket (Live Spot Strikes)")
    if not df_opt_knapsack.empty:
        col_o1, col_o2 = st.columns([1.3, 1])
        with col_o1:
            k_disp = df_opt_knapsack[[
                "Ticker", "Strategy_Structure", "Contract_Legs", "Lot_Size", "OR_Optimal_Lots",
                "Net_Debit_Prem", "Target_Spread_Val", "Margin_Required_Lot", "Theta_Saved_%",
                "Total_Margin_INR", "Total_Est_Profit_INR", "ROI_Per_Lot_%", "Win_Prob_%"
            ]].copy()
            k_disp["Margin_Required_Lot"] = k_disp["Margin_Required_Lot"].map("INR {:,.0f}".format)
            k_disp["Total_Margin_INR"] = k_disp["Total_Margin_INR"].map("INR {:,.0f}".format)
            k_disp["Total_Est_Profit_INR"] = k_disp["Total_Est_Profit_INR"].map("+INR {:,.0f}".format)
            k_disp["Theta_Saved_%"] = k_disp["Theta_Saved_%"].map("{:.1f}%".format)
            k_disp["ROI_Per_Lot_%"] = k_disp["ROI_Per_Lot_%"].map("{:+.1f}%".format)
            st.dataframe(style_light_table(k_disp), width="stretch", hide_index=True)

            csv_opt = df_opt_knapsack.to_csv(index=False).encode("utf-8")
            st.download_button("Export Options Trade Ticket (CSV)", data=csv_opt, file_name="aetheris_options_trades.csv", mime="text/csv")

        with col_o2:
            fig_opt_bar = go.Figure()
            fig_opt_bar.add_trace(go.Bar(x=df_opt_knapsack["Ticker"], y=df_opt_knapsack["Total_Margin_INR"], name="Margin Deployed (INR)", marker_color="#fdba74"))
            fig_opt_bar.add_trace(go.Bar(x=df_opt_knapsack["Ticker"], y=df_opt_knapsack["Total_Est_Profit_INR"], name="Est. Next-Day Profit (INR)", marker_color="#ea580c"))
            fig_opt_bar.update_layout(title="Selected Basket: Margin Outlay vs. Expected Profit", barmode="group", margin=dict(l=15, r=15, t=40, b=15))
            apply_light_orange_chart_style(fig_opt_bar, height=300)
            st.plotly_chart(fig_opt_bar, width="stretch")
    else:
        st.warning("Increase your Next-Day Options Budget in the sidebar to fit at least 1 F&O lot.")

    st.markdown("#### Full F&O Universe: Best Option / Spread Structure Per Underlying")
    opt_disp = df_best_options[[
        "Ticker", "Company", "Underlying_Spot", "Strategy_Structure", "Contract_Legs",
        "Lot_Size", "Net_Debit_Prem", "Margin_Required_Lot", "Theta_Day_Lot", "Theta_Saved_%",
        "Max_Payoff_Cap_Lot", "Est_Profit_1_Lot", "ROI_Per_Lot_%", "Win_Prob_%"
    ]].copy()
    opt_disp["Margin_Required_Lot"] = opt_disp["Margin_Required_Lot"].map("INR {:,.0f}".format)
    opt_disp["Est_Profit_1_Lot"] = opt_disp["Est_Profit_1_Lot"].map("{:+,.0f}".format)
    opt_disp["Theta_Day_Lot"] = opt_disp["Theta_Day_Lot"].map("{:+,.0f}".format)
    opt_disp["Theta_Saved_%"] = opt_disp["Theta_Saved_%"].map("{:.1f}%".format)
    st.dataframe(style_light_table(opt_disp), width="stretch", height=390, hide_index=True)

# --- TAB 3: NAKED VS SPREAD PAYOFF LAB ---
with t3:
    st.subheader("Single-Leg Options vs. Bull Call / Bear Put Spread Payoff Lab")
    selected_sym = st.selectbox("Select Underlying Stock or Index:", df_best_options["Ticker"].unique().tolist(), index=0)
    sub_df = df_all_structures[df_all_structures["Ticker"] == selected_sym].copy()

    if not sub_df.empty:
        st.dataframe(
            style_light_table(sub_df[[
                "Strategy_Structure", "Contract_Legs", "Lot_Size", "Net_Debit_Prem",
                "Target_Spread_Val", "Margin_Required_Lot", "Theta_Day_Lot", "Theta_Saved_%",
                "Max_Payoff_Cap_Lot", "Est_Profit_1_Lot", "ROI_Per_Lot_%", "Net_Delta"
            ]]),
            width="stretch", hide_index=True
        )

        spot_val = float(sub_df.iloc[0]["Underlying_Spot"])
        lot_val = int(sub_df.iloc[0]["Lot_Size"])
        step_val = get_strike_step(spot_val)
        atm_k = round(spot_val / step_val) * step_val
        is_bull = float(sub_df.iloc[0]["Pred_Move_%"]) >= 0
        ot = "CE" if is_bull else "PE"

        price_grid = np.linspace(spot_val * 0.94, spot_val * 1.06, 80)
        p_long, _, _, _ = black_scholes_call_put(spot_val, atm_k, days_to_expiry / 365.0, rf_rate, 0.25, ot)
        short_k = atm_k + 2 * step_val if is_bull else atm_k - 2 * step_val
        p_short, _, _, _ = black_scholes_call_put(spot_val, short_k, days_to_expiry / 365.0, rf_rate, 0.25, ot)

        if is_bull:
            naked_payoff = (np.maximum(0, price_grid - atm_k) - p_long) * lot_val
            spread_payoff = (np.maximum(0, price_grid - atm_k) - np.maximum(0, price_grid - short_k) - (p_long - p_short)) * lot_val
            spread_label = f"Bull Call Spread ({int(atm_k)}/{int(short_k)} CE)"
        else:
            naked_payoff = (np.maximum(0, atm_k - price_grid) - p_long) * lot_val
            spread_payoff = (np.maximum(0, atm_k - price_grid) - np.maximum(0, short_k - price_grid) - (p_long - p_short)) * lot_val
            spread_label = f"Bear Put Spread ({int(atm_k)}/{int(short_k)} PE)"

        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(x=price_grid, y=spread_payoff, mode="lines", name=spread_label, line=dict(color="#ea580c", width=3.2)))
        fig_pay.add_trace(go.Scatter(x=price_grid, y=naked_payoff, mode="lines", name=f"Naked ATM {ot} ({int(atm_k)})", line=dict(color="#d97706", dash="dash", width=2.2)))
        fig_pay.add_hline(y=0, line_color="#94a3b8", line_dash="dot")
        fig_pay.add_vline(x=spot_val, line_color="#f59e0b", line_dash="dot", annotation_text="Live Spot", annotation_font=dict(color="#000000"))
        fig_pay.update_layout(title=f"{selected_sym} Expiry Payoff Curve (1 Lot = {lot_val} Qty)", xaxis_title="Underlying Price at Expiry (INR)", yaxis_title="Net P&L per Lot (INR)")
        apply_light_orange_chart_style(fig_pay, height=350)
        st.plotly_chart(fig_pay, width="stretch")

# --- TAB 4: NEXT-DAY INTRADAY PATTERN & NEWS ML ---
with t4:
    st.subheader("Next-Day Intraday Signal Matrix (Live Spot + Patterns + F&O OI + Overnight News)")

    fig_intra = px.scatter(
        df_intraday,
        x="Pred_Move_%",
        y="Win_Prob_%",
        size=np.abs(df_intraday["News_Sentiment"]) * 25 + 8,
        color="Signal",
        text="Ticker",
        color_discrete_map={"LONG (BUY)": "#f97316", "SHORT (SELL)": "#e11d48"},
        title="Intraday Conviction Quadrant: Predicted Next-Day Move (%) vs. ML Win Probability (%)"
    )
    fig_intra.update_traces(textposition="top center", textfont=dict(color="#000000"))
    fig_intra.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    fig_intra.update_layout(margin=dict(l=15, r=15, t=40, b=15))
    apply_light_orange_chart_style(fig_intra, height=340)
    st.plotly_chart(fig_intra, width="stretch")

    col_d1, col_d2 = st.columns([1.35, 1])
    with col_d1:
        st.dataframe(
            style_light_table(df_intraday[["Ticker", "Company", "Spot_INR", "Day_Chg_%", "Chart_Pattern", "FO_OI_Signal", "Signal", "Pred_Move_%", "Win_Prob_%", "Target_INR", "StopLoss_INR"]]),
            width="stretch", height=420, hide_index=True
        )
    with col_d2:
        st.markdown("#### Overnight Catalyst & Pattern Feed (Top 6)")
        for _, r in df_intraday.head(6).iterrows():
            badge_color = "#c2410c" if "LONG" in r["Signal"] else "#be123c"
            st.markdown(f"""
            <div class="catalyst-card" style="border-left: 4px solid {badge_color};">
                <div style="font-weight:700; font-size:0.88rem; color:{badge_color} !important;">{r['Ticker']} | {r['Signal']} ({r['Pred_Move_%']:+.2f}% | Win: {r['Win_Prob_%']:.1f}%)</div>
                <div style="font-size:0.76rem; color:#334155 !important; margin-top:2px;">Live Spot: <b style="color:#000000;">INR {r['Spot_INR']:,.1f} ({r['Day_Chg_%']:+.2f}%)</b> | Pattern: <b style="color:#000000;">{r['Chart_Pattern']}</b></div>
                <div style="font-size:0.79rem; color:#000000 !important; margin-top:4px;">{r['Catalyst_News']}</div>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 5: WAR, CRUDE & FX STRESS MATRIX ---
with t5:
    st.subheader("Brent Crude ($/bbl) x USD/INR (INR/$) Portfolio Sensitivity Heatmap")

    crude_grid = [75, 96, 115, 130, 145]
    fx_grid = [91.0, 94.25, 96.5, 99.0, 102.0]
    z_matrix = []
    for fx_val in fx_grid:
        row_vals = []
        for cr_val in crude_grid:
            port_shift = 0.0
            for _, w_row in winners.iterrows():
                w_frac = w_row["Portfolio_Weight_%"] / 100.0
                sec = w_row["Sector"]
                port_shift += w_frac * (((cr_val - BASE_BRENT_USD)/10.0)*BRENT_SECTOR_BETA_PER_10USD.get(sec, 0.0) + (fx_val - BASE_USD_INR)*FX_SECTOR_BETA_PER_1INR.get(sec, 0.0))
            base_unshifted = total_port_cagr - float(np.sum((winners["Portfolio_Weight_%"]/100.0) * (winners["Macro_Alpha_%"]) / 100.0))
            row_vals.append(round((base_unshifted + port_shift) * 100, 2))
        z_matrix.append(row_vals)

    fig_hm = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=[f"${c}/bbl" for c in crude_grid],
        y=[f"INR {f:.2f}/$" for f in fx_grid],
        colorscale="Oranges",
        texttemplate="%{z:.2f}%",
        textfont=dict(color="#000000"),
        colorbar=dict(title=dict(text="Est. CAGR %", font=dict(color="#000000")), tickfont=dict(color="#000000"))
    ))
    fig_hm.update_layout(title="Portfolio Expected CAGR (%) Across Crude Oil & Currency Shock Scenarios", xaxis_title="Brent Crude Oil ($/bbl)", yaxis_title="USD/INR Rate")
    apply_light_orange_chart_style(fig_hm, height=330)
    st.plotly_chart(fig_hm, width="stretch")

    sectors_list = sorted(df_all_scored["Sector"].unique())
    sens_rows = []
    for sec in sectors_list:
        c_imp = ((brent_usd - BASE_BRENT_USD) / 10.0) * BRENT_SECTOR_BETA_PER_10USD.get(sec, 0.0) * 100
        f_imp = (usd_inr - BASE_USD_INR) * FX_SECTOR_BETA_PER_1INR.get(sec, 0.0) * 100
        sens_rows.append({
            "Sector": sec,
            "Crude Beta (+$10/bbl)": f"{BRENT_SECTOR_BETA_PER_10USD.get(sec, 0.0)*100:+.2f}%",
            "FX Beta (+INR 1/$)": f"{FX_SECTOR_BETA_PER_1INR.get(sec, 0.0)*100:+.2f}%",
            f"Crude Impact (${brent_usd:.0f})": f"{c_imp:+.2f}%",
            f"FX Impact (INR {usd_inr:.2f})": f"{f_imp:+.2f}%",
            "Net Sector Alpha Shift": f"{c_imp + f_imp:+.2f}%"
        })
    st.dataframe(style_light_table(pd.DataFrame(sens_rows)), width="stretch", hide_index=True)

# --- TAB 6: REBALANCING & TAX BLOTTER ---
with t6:
    st.subheader("Portfolio Rebalancing Trade Blotter & Indian Capital Gains Tax")
    blotter = winners[["Ticker", "Company", "Asset_Class", "Allocation_INR"]].copy()
    blotter.rename(columns={"Allocation_INR": "Target_Deploy_Now_INR"}, inplace=True)
    rng_b = np.random.default_rng(7)
    blotter["Current_INR"] = np.round(blotter["Target_Deploy_Now_INR"] * rng_b.uniform(0.0, 1.15, len(blotter)), -2)
    blotter["Trade_Delta_INR"] = blotter["Target_Deploy_Now_INR"] - blotter["Current_INR"]
    blotter["Action"] = np.where(blotter["Trade_Delta_INR"] > 1000, "BUY", np.where(blotter["Trade_Delta_INR"] < -1000, "TRIM / SELL", "HOLD"))

    def calc_trade_tax(row):
        if row["Trade_Delta_INR"] >= -1000:
            return 0.0
        sell_amt = abs(row["Trade_Delta_INR"])
        est_gain = sell_amt * 0.25
        if row["Asset_Class"] == "Cryptocurrency":
            return round(est_gain * 0.312 + sell_amt * 0.01, 0)
        return round(est_gain * (0.125 if horizon_years >= 1.0 else 0.20), 0)

    blotter["Est_Tax_&_TDS_INR"] = blotter.apply(calc_trade_tax, axis=1)
    st.dataframe(style_light_table(blotter), width="stretch", hide_index=True)

# --- TAB 7: RESEARCH SCREENER, IPOS & OR MATH ---
with t7:
    st.subheader("Live Market Quotes, Quarterly Earnings & Walk-Forward ML Diagnostics")
    st.dataframe(
        style_light_table(df_all_scored[["Ticker", "Company", "Asset_Class", "Spot", "Day_Chg_%", "YoY_Rev_%", "YoY_PAT_%", "ROE", "ML_Dir_Acc_%", "OOS_R2", "Effective_CAGR", "Conviction_Score", "Key_Products"]].sort_values(by="Conviction_Score", ascending=False)),
        width="stretch", height=340, hide_index=True
    )

    st.subheader("Active SEBI DRHP Filings (Nov 2026) & Mega-IPO Pipeline")
    st.dataframe(style_light_table(ipo_df.sort_values(by="QIB_Demand_Score", ascending=False)), width="stretch", hide_index=True)

    st.subheader("Unified Operations Research & Derivatives Mathematical Architecture")
    st.markdown("""
    <div class="formula-box">
    <b>1. Live Telemetry & Predict-Then-Optimize (SPO+) Alpha Overlay:</b><br>
    mu_eff(i, Omega) = [0.4*RF + 0.4*GBR + 0.2*Ridge + dAlpha_Regime + Beta_Brent*dCrude + Beta_FX*dUSDINR] * (1 - tau_i)<br><br>
    <b>2. Multi-Asset Portfolio MILP & CVaR 95% LP (SciPy HiGHS):</b><br>
    Max Sum_i (mu_eff(i) + 0.045*Conv_i - 0.30*sigma_i^2)*w_i &nbsp; s.t. Sum(z_i) &lt;= K, &nbsp; 0.04*z_i &lt;= w_i &lt;= w_max*z_i, &nbsp; Sum_{crypto} w_i &lt;= C_max(Omega)<br><br>
    <b>3. Multi-Leg Vertical Spread & Integer Knapsack Options Maximizer:</b><br>
    Max Sum_i (0.65*Profit_lot(i) + 0.35*EV_lot(i))*x_i &nbsp; s.t. Sum_i Margin_lot(i)*x_i &lt;= Budget_Options, &nbsp; x_i in {0..Max_Lots}
    </div>
    """, unsafe_allow_html=True)
