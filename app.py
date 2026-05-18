"""
app.py — SmartStartup AI: Gamified, Explainable & Generative Startup Success Prediction System
Main Streamlit entry point.
"""

import html
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import ConfusionMatrixDisplay
import io

from preprocessing import preprocess, FEATURE_COLS, FEATURE_DISPLAY_NAMES
from model import train_models, predict_single, save_models, load_models
from xai import get_shap_explainer, get_shap_values_single, plot_shap_bar
from generative import (get_business_explanation, get_strategy_advice,
                         get_whatif_explanation, get_gamified_feedback)
from currency import (format_currency, convert_to_usd, convert_from_usd,
                       get_currency_options, CURRENCY_NAMES, CURRENCY_SYMBOLS,
                       EXCHANGE_RATES)
from pakistan_data import get_pakistan_df
from startup_data import build_combined_startup_dataset

# ─────────────────────────── Page config ────────────────────────────
st.set_page_config(
    page_title="SmartStartup AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Syne:wght@600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp {
    background:
      radial-gradient(circle at 18% 8%, rgba(124, 58, 237, 0.16), transparent 34%),
      radial-gradient(circle at 88% 12%, rgba(6, 182, 212, 0.12), transparent 28%),
      linear-gradient(180deg, #070A13 0%, #0D1020 52%, #101522 100%);
    color: #EEF3FF;
  }
  section[data-testid="stSidebar"] {
    background: rgba(9, 13, 24, 0.98);
    border-right: 1px solid rgba(148, 163, 184, 0.16);
  }
  section[data-testid="stSidebar"] * { color: #E8EEF9; }
  [data-testid="stHeader"] { background: rgba(7, 10, 19, 0.76); }
  [data-testid="collapsedControl"] {
    top: 18px;
    left: 18px;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.95), rgba(6, 182, 212, 0.82)) !important;
    border: 1px solid rgba(255, 255, 255, 0.22);
    box-shadow: 0 14px 34px rgba(6, 182, 212, 0.22);
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 1 !important;
  }
  [data-testid="collapsedControl"] svg {
    color: #FFFFFF !important;
    stroke: #FFFFFF !important;
    width: 1.45rem;
    height: 1.45rem;
    opacity: 1 !important;
  }
  [data-testid="collapsedControl"]:hover {
    transform: translateX(2px);
    box-shadow: 0 16px 40px rgba(124, 58, 237, 0.30);
  }
  [data-testid="collapsedControl"]::after {
    content: "Menu";
    position: absolute;
    left: 52px;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(15, 23, 42, 0.96);
    color: #F8FAFC;
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 999px;
    padding: 0.32rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    pointer-events: none;
  }
  button[data-testid="stExpandSidebarButton"] {
    width: 44px !important;
    height: 44px !important;
    border-radius: 12px !important;
    background: rgba(124, 58, 237, 0.95) !important;
    border: 1px solid rgba(255, 255, 255, 0.28) !important;
    box-shadow: 0 12px 30px rgba(124, 58, 237, 0.35), 0 0 0 1px rgba(6, 182, 212, 0.22) !important;
    opacity: 1 !important;
  }
  button[data-testid="stExpandSidebarButton"]:hover {
    background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
    transform: translateX(2px);
  }
  button[data-testid="stExpandSidebarButton"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    opacity: 1 !important;
  }
  button[data-testid="stExpandSidebarButton"] svg {
    width: 1.55rem !important;
    height: 1.55rem !important;
    filter: drop-shadow(0 0 5px rgba(255, 255, 255, 0.45));
  }
  div[data-testid="stMarkdownContainer"] p,
  div[data-testid="stMarkdownContainer"] li { color: #C7D2E5; }
  h1, h2, h3, h4 {
    color: #F8FAFC;
    letter-spacing: 0;
    font-family: 'Syne', sans-serif;
  }

  .main-header {
    background:
      linear-gradient(135deg, rgba(13, 18, 34, 0.98), rgba(23, 22, 48, 0.94)),
      linear-gradient(90deg, rgba(124, 58, 237, 0.20), rgba(6, 182, 212, 0.12));
    padding: 1.8rem 2.2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: #F8FBFF;
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.24);
  }
  .main-header h1 { font-size: 2.15rem; font-weight: 800; margin: 0; letter-spacing: 0; }
  .main-header p  { font-size: 1.05rem; opacity: 0.85; margin: 0.3rem 0 0; }

  .metric-card {
    background: rgba(15, 23, 42, 0.86);
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.18);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-left: 4px solid #8B5CF6;
    text-align: center;
  }
  .metric-card h2 { font-size: 2rem; font-weight: 700; margin: 0; color: #FFFFFF; }
  .metric-card p  { font-size: 0.85rem; color: #A8B3C7; margin: 0; }

  .success-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.20), rgba(15, 23, 42, 0.94));
    color: #D7FFF0; border-radius: 12px; padding: 1.5rem; text-align: left;
    border: 1px solid rgba(16, 185, 129, 0.36);
    box-shadow: 0 16px 44px rgba(16, 185, 129, 0.10);
  }
  .failure-card {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.20), rgba(15, 23, 42, 0.94));
    color: #FFE2E2; border-radius: 12px; padding: 1.5rem; text-align: left;
    border: 1px solid rgba(244, 63, 94, 0.36);
    box-shadow: 0 16px 44px rgba(244, 63, 94, 0.10);
  }
  .prediction-label { font-size: 2.2rem; font-weight: 700; }
  .prediction-sub   { font-size: 1rem; opacity: 0.9; }

  .insight-box {
    background: rgba(15, 23, 42, 0.86);
    border: 1px solid rgba(6, 182, 212, 0.22);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    color: #DCEBFA;
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.16);
  }
  .strategy-box {
    background: rgba(17, 24, 39, 0.9);
    border: 1px solid rgba(139, 92, 246, 0.24);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    color: #F2EAF4;
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.16);
  }
  .game-card {
    background: linear-gradient(135deg, rgba(9, 18, 32, 0.96), rgba(27, 38, 59, 0.9));
    color: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.16);
  }
  .game-card h3 { color: #ffffff; font-size: 1.3rem; margin: 0 0 0.5rem; }

  .currency-badge {
    background: rgba(139, 92, 246, 0.14); color: #DDD6FE;
    border: 1px solid rgba(139, 92, 246, 0.28);
    border-radius: 999px; padding: 0.2rem 0.8rem;
    font-size: 0.85rem; font-weight: 600; display: inline-block;
  }
  .pak-badge {
    background: rgba(52, 211, 153, 0.14); color: #CFFFE8;
    border: 1px solid rgba(52, 211, 153, 0.28);
    border-radius: 999px; padding: 0.2rem 0.8rem;
    font-size: 0.85rem; font-weight: 600; display: inline-block;
  }

  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
  }
  div[data-testid="stMetric"] {
    background: rgba(12, 20, 36, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 12px;
    padding: 0.5rem;
  }
  .result-section {
    margin-top: 1.4rem;
    padding: 1.2rem;
    background: rgba(9, 13, 24, 0.46);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
  }
  .soft-panel {
    padding: 1rem;
    background: rgba(15, 23, 42, 0.74);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    margin-bottom: 1rem;
  }
  div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    overflow: hidden;
  }
  .factor-table {
    width: 100%;
    border-collapse: collapse;
    overflow: hidden;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.16);
  }
  .factor-table th {
    text-align: left;
    color: #F8FAFC;
    background: rgba(124, 58, 237, 0.20);
    font-size: 0.82rem;
    padding: 0.75rem;
  }
  .factor-table td {
    color: #C7D2E5;
    border-top: 1px solid rgba(148, 163, 184, 0.10);
    padding: 0.75rem;
    vertical-align: top;
    font-size: 0.9rem;
  }
  .factor-table td:first-child { color: #FFFFFF; font-weight: 700; }
  .model-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 10px;
  }
  .model-table th {
    color: #94A3B8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: left;
    padding: 0.5rem 0.85rem;
  }
  .model-table td {
    background: rgba(15, 23, 42, 0.86);
    color: #DCEBFA;
    padding: 0.95rem 0.85rem;
    border-top: 1px solid rgba(148, 163, 184, 0.14);
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
  }
  .model-table td:first-child {
    border-left: 4px solid #8B5CF6;
    border-radius: 12px 0 0 12px;
    color: #FFFFFF;
    font-weight: 800;
  }
  .model-table td:last-child {
    border-right: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 0 12px 12px 0;
  }
  .score-pill {
    display: inline-block;
    min-width: 58px;
    text-align: center;
    padding: 0.28rem 0.58rem;
    border-radius: 999px;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.24);
    color: #A5F3FC;
    font-weight: 800;
  }
  .section-kicker {
    color: #A78BFA;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 800;
    margin-bottom: 0.35rem;
  }
  div.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(6, 182, 212, 0.28);
    background: linear-gradient(135deg, #7C3AED, #06B6D4);
    color: white;
    font-weight: 800;
    box-shadow: 0 14px 32px rgba(124, 58, 237, 0.22);
  }
  div[data-baseweb="input"],
  div[data-baseweb="select"] > div,
  div[data-baseweb="textarea"],
  input,
  textarea {
    background: rgba(8, 13, 26, 0.86) !important;
    border-color: rgba(148, 163, 184, 0.24) !important;
    color: #F8FAFC !important;
    border-radius: 10px !important;
  }
  div[data-baseweb="select"] span,
  div[data-baseweb="select"] div,
  input::placeholder {
    color: #A8B3C7 !important;
  }
  div[data-testid="stSlider"] [role="slider"] {
    background: #06B6D4 !important;
    border-color: #06B6D4 !important;
  }
  div[data-testid="stSlider"] div[data-testid="stTickBar"] {
    background: rgba(148, 163, 184, 0.22) !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label {
    min-height: 42px;
    border-radius: 10px;
    padding: 0 12px !important;
    background: transparent;
    border: 1px solid transparent;
    transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(148, 163, 184, 0.08);
    border-color: rgba(148, 163, 184, 0.14);
    transform: translateX(2px);
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(124, 58, 237, 0.16);
    border-color: rgba(124, 58, 237, 0.32);
    box-shadow: inset 3px 0 0 #8B5CF6;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] p {
    font-weight: 700;
    color: #DCEBFA !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── Session State Init ──────────────────────────
def render_text_box(text: str, css_class: str = "insight-box"):
    """Render model-generated text safely inside a styled box."""
    safe_text = html.escape(str(text)).replace("\n", "<br>")
    st.markdown(f'<div class="{css_class}">{safe_text}</div>', unsafe_allow_html=True)


def render_factor_table(rows):
    if not rows:
        st.caption("No strong factors in this group.")
        return

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['Factor']))}</td>"
            f"<td>{html.escape(str(row['What it means']))}</td>"
            f"<td>{html.escape(str(row['Effect']))}</td>"
            f"<td>{html.escape(str(row['Impact']))}</td>"
            "</tr>"
        )
    st.markdown(
        """
        <table class="factor-table">
          <thead>
            <tr><th>Factor</th><th>What it means</th><th>Effect</th><th>Impact</th></tr>
          </thead>
          <tbody>
        """
        + "".join(table_rows)
        + """
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_model_table(rows):
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['Model']))}</td>"
            f"<td><span class='score-pill'>{html.escape(str(row.get('Balanced Accuracy', row['Accuracy'])))}</span></td>"
            f"<td>{html.escape(str(row['Accuracy']))}</td>"
            f"<td>{html.escape(str(row['Precision']))}</td>"
            f"<td>{html.escape(str(row.get('Success Recall', row.get('Recall', ''))))}</td>"
            f"<td>{html.escape(str(row.get('Failure Recall', '')))}</td>"
            f"<td>{html.escape(str(row['F1-Score']))}</td>"
            f"<td>{html.escape(str(row['ROC-AUC']))}</td>"
            "</tr>"
        )
    st.markdown(
        """
        <table class="model-table">
          <thead>
            <tr>
              <th>Model</th><th>Balanced Acc.</th><th>Accuracy</th><th>Precision</th><th>Success Recall</th><th>Failure Recall</th><th>F1 Score</th><th>ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
        """
        + "".join(body)
        + """
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def create_probability_gauge(probability: float, is_success: bool, height: int = 210):
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Success Probability", "font": {"size": 14, "color": "#DCEBFA"}},
        number={"suffix": "%", "font": {"size": 24, "color": "#FFFFFF"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#9FB2C8",
                "tickfont": {"color": "#C7D2E5", "size": 11},
            },
            "bar": {"color": "#34D399" if is_success else "#FB7185", "thickness": 0.32},
            "bgcolor": "rgba(12,20,36,0.36)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(251,113,133,0.22)"},
                {"range": [40, 65], "color": "rgba(251,191,36,0.18)"},
                {"range": [65, 100], "color": "rgba(52,211,153,0.18)"},
            ],
            "threshold": {"line": {"color": "#CBD5E1", "width": 2}, "value": 50}
        }
    ))
    fig_gauge.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#DCEBFA",
        margin=dict(t=34, b=0, l=8, r=8),
    )
    return fig_gauge


FEATURE_USER_LABELS = {
    "Total Funding (log)": "Funding strength",
    "# Funding Rounds": "Investor traction",
    "Startup Age (log years)": "Operating maturity",
    "Funding Duration (log days)": "Funding continuity",
    "Days to First Funding": "Speed to first investment",
    "Country": "Market environment",
    "Region": "Regional ecosystem",
    "Industry Category": "Industry fit",
    "Pakistan-based": "Pakistan market signal",
}

FEATURE_USER_MEANINGS = {
    "Total Funding (log)": "How much capital the startup has raised so far.",
    "# Funding Rounds": "Whether investors have backed the company across multiple rounds.",
    "Startup Age (log years)": "How long the company has had to build operations and market proof.",
    "Funding Duration (log days)": "Whether funding activity continued over time instead of stopping early.",
    "Days to First Funding": "How quickly the startup attracted its first outside investment.",
    "Country": "The broader business and funding conditions in the selected market.",
    "Region": "The local ecosystem around the startup.",
    "Industry Category": "How the selected sector performs in the training data.",
    "Pakistan-based": "The model signal from Pakistan startup examples in the dataset.",
}


def build_factor_tables(shap_values, feature_names):
    """Convert SHAP values into user-friendly factor tables and text for Groq."""
    rows = []
    for feature, value in zip(feature_names, shap_values):
        direction = "Helps prediction" if value >= 0 else "Needs attention"
        strength = abs(float(value))
        if strength >= 1:
            impact = "High"
        elif strength >= 0.25:
            impact = "Medium"
        else:
            impact = "Low"
        rows.append({
            "Factor": FEATURE_USER_LABELS.get(feature, feature),
            "What it means": FEATURE_USER_MEANINGS.get(feature, "A model signal used in the prediction."),
            "Effect": direction,
            "Impact": impact,
            "Model score": round(float(value), 3),
        })

    positive = sorted([r for r in rows if r["Model score"] >= 0], key=lambda r: r["Model score"], reverse=True)[:4]
    risk = sorted([r for r in rows if r["Model score"] < 0], key=lambda r: r["Model score"])[:4]
    pos_text = "\n".join([f"- {r['Factor']}: {r['What it means']} ({r['Impact']} positive impact)" for r in positive])
    risk_text = "\n".join([f"- {r['Factor']}: {r['What it means']} ({r['Impact']} risk signal)" for r in risk])
    return pd.DataFrame(positive), pd.DataFrame(risk), pos_text, risk_text


if "model_data" not in st.session_state:
    st.session_state.model_data = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "currency" not in st.session_state:
    st.session_state.currency = "USD"
if "game_score" not in st.session_state:
    st.session_state.game_score = 0
if "game_round" not in st.session_state:
    st.session_state.game_round = 0


# ─────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.markdown("## SmartStartup AI")
    st.markdown("---")

    # Currency Selector
    st.markdown("### Currency Settings")
    currency_options = get_currency_options()
    selected_currency = st.selectbox(
        "Display Currency",
        currency_options,
        index=currency_options.index(st.session_state.currency),
        format_func=lambda c: f"{CURRENCY_SYMBOLS[c]} {CURRENCY_NAMES[c]}",
        key="currency_selector"
    )
    st.session_state.currency = selected_currency
    CURR = selected_currency

    if CURR == "PKR":
        st.markdown('<span class="pak-badge">Pakistan Market Mode</span>', unsafe_allow_html=True)
        rate = EXCHANGE_RATES["PKR"]
        st.caption(f"1 USD = ₨{rate:,.1f} PKR")
    else:
        st.markdown(f'<span class="currency-badge">{CURRENCY_SYMBOLS[CURR]} {CURR}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Model Selection
    st.markdown("### Model")
    model_choice = st.selectbox(
        "Select Model",
        ["XGBoost", "Random Forest", "Logistic Regression"],
        help="XGBoost gives best accuracy. Logistic Regression is most interpretable."
    )

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to", [
        "Dashboard",
        "Predict Startup",
        "What-If Analysis",
        "Startup Challenge",
        "Model Performance",
        "Pakistan Startups",
    ])

    st.markdown("---")
    st.caption("SmartStartup AI v1.0 | ML, explainability, and Groq analysis")


# ─────────────────────── Load / Train Model ──────────────────────────
@st.cache_resource(show_spinner=False)
def get_trained_models():
    """Train from the uploaded Crunchbase dataset plus Pakistan augmentation."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    combined_path = os.path.join(data_dir, "combined_startup_dataset.csv")
    combined, training_df = build_combined_startup_dataset(
        pakistan_share=0.25,
        save_path=combined_path,
    )

    processed, le_c, le_r, le_cat = preprocess(training_df)
    processed = processed.dropna(subset=["target"]).copy()
    X = processed[FEATURE_COLS].values
    y = processed["target"].values.astype(int)

    trained, scaler, results, X_test, y_test = train_models(X, y)
    return {"models": trained, "scaler": scaler, "results": results,
            "le_country": le_c, "le_region": le_r, "le_cat": le_cat,
            "df": combined, "df_processed": processed,
            "X_test": X_test, "y_test": y_test,
            "combined_path": combined_path}


# Show loading spinner for model
with st.spinner("Initialising SmartStartup AI: training models on Crunchbase + Pakistan data..."):
    md = get_trained_models()

models_dict = md["models"]
scaler = md["scaler"]
results = md["results"]
le_country = md["le_country"]
le_region = md["le_region"]
le_cat = md["le_cat"]
df_all = md["df"]
df_proc = md["df_processed"]
X_test = md["X_test"]
y_test = md["y_test"]
CURR = st.session_state.currency


# ════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div class="main-header">
      <h1>SmartStartup AI</h1>
      <p>Startup success prediction with explainable machine learning and Groq-powered business analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_all)
    pak_count = len(df_all[df_all.get("country_code", pd.Series([])) == "PAK"]) if "country_code" in df_all else 21
    best_acc = max(r["balanced_accuracy"] for r in results.values())
    best_model_name = max(results, key=lambda k: results[k]["balanced_accuracy"])

    with col1:
        st.markdown(f"""<div class="metric-card">
          <h2>{total:,}</h2><p>Total Startup Records</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
          <h2>{pak_count}</h2><p>Pakistan Startups Added</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
          <h2>{best_acc*100:.1f}%</h2><p>Best Balanced Accuracy</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
          <h2>3</h2><p>ML Models Trained</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Feature highlights
    st.subheader("System Capabilities")
    c1, c2, c3, c4 = st.columns(4)
    caps = [
        ("", "ML Prediction", "Logistic Regression, Random Forest and XGBoost trained on 3,000+ startup records including Pakistan data"),
        ("", "Explainable AI", "SHAP-powered feature importance shows why each prediction was made"),
        ("", "Groq Analysis", "Groq converts predictions into plain-language business insights and strategy advice"),
        ("", "Startup Challenge", "Game-style startup scenario with budget allocation and AI feedback"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], caps):
        with col:
            st.markdown(f"### {title}")
            st.caption(desc)

    st.markdown("---")
    st.subheader(f"Model Performance Overview ({CURRENCY_NAMES[CURR]})")

    model_perf = pd.DataFrame({
        "Model": list(results.keys()),
        "Balanced Accuracy": [f"{r['balanced_accuracy']*100:.1f}%" for r in results.values()],
        "Accuracy": [f"{r['accuracy']*100:.1f}%" for r in results.values()],
        "Precision": [f"{r['precision']*100:.1f}%" for r in results.values()],
        "Success Recall": [f"{r['recall']*100:.1f}%" for r in results.values()],
        "Failure Recall": [f"{r['failure_recall']*100:.1f}%" for r in results.values()],
        "F1-Score": [f"{r['f1']*100:.1f}%" for r in results.values()],
        "ROC-AUC": [f"{r['roc_auc']:.3f}" for r in results.values()],
    })
    render_model_table(model_perf.to_dict("records"))

    # Currency info
    st.info(f"All financial figures displayed in **{CURRENCY_NAMES[CURR]}** "
            f"(1 USD = {EXCHANGE_RATES[CURR]:,.2f} {CURR}). Change currency in the sidebar.")


# ════════════════════════════════════════════════════
#  PAGE: PREDICT STARTUP
# ════════════════════════════════════════════════════
elif page == "Predict Startup":
    st.markdown("""<div class="main-header">
      <h1>Startup Success Predictor</h1>
      <p>Enter a startup profile to generate a prediction, explanation, and strategy recommendations</p>
    </div>""", unsafe_allow_html=True)

    sym = CURRENCY_SYMBOLS[CURR]
    rate = EXCHANGE_RATES[CURR]

    form_col, instant_col = st.columns([1.15, 0.85], gap="large")

    with form_col:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.subheader("Startup Profile")

        startup_name = st.text_input("Startup Name", placeholder="e.g. TechVenture PK")

        industry_opts = ["Software", "FinTech", "E-Commerce", "HealthTech", "EdTech",
                          "Logistics", "B2B Commerce", "Transportation", "Media & Entertainment",
                          "Real Estate", "Analytics & Data", "Cloud Computing", "Mobile",
                          "Other"]
        category = st.selectbox("Industry / Category", industry_opts)

        # Country — Pakistan highlighted
        country_opts = ["PAK - Pakistan", "USA - United States", "GBR - United Kingdom",
                         "IND - India", "ARE - UAE", "SAR - Saudi Arabia",
                         "DEU - Germany", "CAN - Canada", "SGP - Singapore",
                         "AUS - Australia", "FRA - France", "CHN - China", "Other"]
        country_sel = st.selectbox("Country", country_opts, index=0)
        country_code = country_sel.split(" - ")[0]

        region_opts = ["Sindh (Karachi)", "Punjab (Lahore)", "Islamabad", "KPK (Peshawar)",
                        "Balochistan", "Gilgit-Baltistan",
                        "California", "New York", "London", "Dubai", "Other"]
        region = st.selectbox("Region / Province", region_opts)

        # Funding in selected currency
        st.markdown(f"**Total Funding Raised ({sym} {CURR})**")
        max_fund_local = int(convert_from_usd(500_000_000, CURR))
        step_local = max(1000, int(convert_from_usd(10_000, CURR)))
        default_fund = int(convert_from_usd(500_000, CURR))

        funding_local = st.number_input(
            f"Amount in {CURR}",
            min_value=0,
            max_value=max_fund_local,
            value=default_fund,
            step=step_local,
            help=f"Enter funding in {CURRENCY_NAMES[CURR]}"
        )
        funding_usd = convert_to_usd(funding_local, CURR)
        st.caption(f"≈ USD ${funding_usd:,.0f}")

        funding_rounds = st.slider("Number of Funding Rounds", 0, 15, 2)
        startup_age = st.slider("Startup Age (years)", 0, 20, 3)
        days_first_funding = st.slider("Days Until First Funding", 0, 1825, 365,
                                        help="How many days after founding did you get first funding?")

        predict_btn = st.button("Run Prediction", type="primary", width="stretch")

        st.markdown('</div>', unsafe_allow_html=True)

    preview_label = None
    preview_prob = None
    if predict_btn:
        try:
            preview_cat_enc = le_cat.transform([category])[0]
        except:
            preview_cat_enc = 0
        try:
            preview_country_clean = country_code if country_code in le_country.classes_ else "USA"
            preview_country_enc = le_country.transform([preview_country_clean])[0]
        except:
            preview_country_enc = 0
        try:
            preview_region_clean = region.split(" (")[0] if "(" in region else region
            preview_region_enc = le_region.transform([preview_region_clean])[0] if preview_region_clean in le_region.classes_ else 0
        except:
            preview_region_enc = 0
        preview_input = np.array([[
            np.log1p(funding_usd), funding_rounds, np.log1p(startup_age),
            np.log1p(0), days_first_funding,
            2024 - startup_age, np.log1p(funding_usd / max(funding_rounds, 1)),
            preview_country_enc, preview_region_enc, preview_cat_enc, int(country_code == "PAK")
        ]])
        preview_label, preview_prob = predict_single(models_dict, scaler, preview_input, model_choice)

    with instant_col:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Live Decision</div>', unsafe_allow_html=True)
        st.markdown("### Prediction Preview")
        if predict_btn:
            if preview_label == 1:
                st.markdown(f"""<div class="success-card">
                  <div class="prediction-label">PREDICTED SUCCESS</div>
                  <div class="prediction-sub">Estimated success probability: {preview_prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="failure-card">
                  <div class="prediction-label">HIGH RISK</div>
                  <div class="prediction-sub">Estimated success probability: {preview_prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            st.plotly_chart(
                create_probability_gauge(preview_prob, preview_label == 1, height=220),
                width="stretch"
            )
            st.markdown(f"""
            <p><strong>{html.escape(startup_name or 'Your Startup')}</strong></p>
            <p>{html.escape(category)} in {html.escape(country_sel)}<br>
            Funding: {html.escape(format_currency(funding_usd, CURR))}<br>
            Rounds: {funding_rounds} | Age: {startup_age} years</p>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <p>Complete the questionnaire and run the model to see the success estimate here first.</p>
            <p>The detailed factors, AI analysis, and recommendations will remain below the form.</p>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Analysing startup profile..."):
            try:
                cat_enc = le_cat.transform([category])[0]
            except:
                cat_enc = 0
            try:
                country_clean = country_code if country_code in le_country.classes_ else "USA"
                country_enc = le_country.transform([country_clean])[0]
            except:
                country_enc = 0
            try:
                region_clean = region.split(" (")[0] if "(" in region else region
                region_enc = le_region.transform([region_clean])[0] if region_clean in le_region.classes_ else 0
            except:
                region_enc = 0

            is_pak = int(country_code == "PAK")
            X_input = np.array([[
                np.log1p(funding_usd), funding_rounds, np.log1p(startup_age),
                np.log1p(0), days_first_funding,
                2024 - startup_age, np.log1p(funding_usd / max(funding_rounds, 1)),
                country_enc, region_enc, cat_enc, is_pak
            ]])

            label, prob = predict_single(models_dict, scaler, X_input, model_choice)
            model_for_shap = models_dict[model_choice]
            X_bg = df_proc[FEATURE_COLS].values[:100]
            explainer = get_shap_explainer(model_for_shap, X_bg, model_choice)
            shap_vals = get_shap_values_single(explainer, X_input, model_choice)
            feat_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLS]
            positive_df, risk_df, pos_txt, neg_txt = build_factor_tables(shap_vals, feat_names)

        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.subheader("Business Factors Behind the Prediction")
        st.caption("The success score is a model estimate, not a guarantee. Use it as a decision-support signal.")
        st.caption("These tables translate the model signals into plain business language.")
        factor_col1, factor_col2 = st.columns(2, gap="large")
        with factor_col1:
            st.markdown("**What is working in your favor**")
            render_factor_table(positive_df.to_dict("records"))
        with factor_col2:
            st.markdown("**What needs attention**")
            render_factor_table(risk_df.to_dict("records"))

        with st.expander("View technical SHAP chart"):
            fig_shap = plot_shap_bar(shap_vals, feat_names)
            st.pyplot(fig_shap, width="stretch")
            plt.close()

        with st.spinner("Generating Groq business analysis..."):
            startup_info = {
                "name": startup_name or "Your Startup",
                "category": category,
                "country": country_sel,
                "funding_display": format_currency(funding_usd, CURR),
                "funding_rounds": funding_rounds,
                "age": startup_age,
            }
            explanation = get_business_explanation(
                startup_info, label, prob, pos_txt, neg_txt, CURR
            )
            strategy = get_strategy_advice(startup_info, label, prob, CURR)

        st.subheader("AI Business Analysis")
        render_text_box(explanation)

        st.subheader("Strategic Recommendations")
        render_text_box(strategy, "strategy-box")
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  PAGE: WHAT-IF ANALYSIS
# ════════════════════════════════════════════════════
elif page == "What-If Analysis":
    st.markdown("""<div class="main-header">
      <h1>What-If Analysis</h1>
      <p>Evaluate how changes in funding, rounds, and market affect the success prediction</p>
    </div>""", unsafe_allow_html=True)

    sym = CURRENCY_SYMBOLS[CURR]

    st.subheader("Base Startup Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        base_funding = st.number_input(
            f"Base Funding ({CURR})",
            value=int(convert_from_usd(500_000, CURR)),
            step=int(convert_from_usd(50_000, CURR))
        )
        base_funding_usd = convert_to_usd(base_funding, CURR)
    with col2:
        base_rounds = st.slider("Base Funding Rounds", 1, 10, 2)
    with col3:
        base_age = st.slider("Base Startup Age (years)", 1, 15, 3)

    base_category = st.selectbox("Industry", ["Software", "FinTech", "E-Commerce", "HealthTech",
                                               "EdTech", "Logistics", "Other"])
    base_country = st.selectbox("Country", ["PAK", "USA", "GBR", "IND", "ARE", "DEU"])

    def build_X(funding_usd, rounds, age, cat, country):
        rounds_safe = max(int(rounds), 1)
        try: cat_enc = le_cat.transform([cat])[0]
        except: cat_enc = 0
        try:
            cc = country if country in le_country.classes_ else "USA"
            country_enc = le_country.transform([cc])[0]
        except: country_enc = 0
        is_pak = int(country == "PAK")
        return np.array([[
            np.log1p(funding_usd), rounds_safe, np.log1p(age),
            np.log1p(0), 365, 2024 - age,
            np.log1p(funding_usd / rounds_safe),
            country_enc, 0, cat_enc, is_pak
        ]])

    X_base = build_X(base_funding_usd, base_rounds, base_age, base_category, base_country)
    base_label, base_prob = predict_single(models_dict, scaler, X_base, model_choice)

    st.markdown(f"**Base Prediction:** {'Success' if base_label == 1 else 'At Risk'} "
                f"— {base_prob*100:.1f}% confidence")

    st.markdown("---")
    st.subheader("Adjust Parameters and Compare")

    tab1, tab2, tab3 = st.tabs(["Funding Impact", "Rounds Impact", "Country Impact"])

    with tab1:
        mult_options = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        probs_funding = []
        labels_x = []
        for m in mult_options:
            new_f = base_funding_usd * m
            Xi = build_X(new_f, base_rounds, base_age, base_category, base_country)
            _, p = predict_single(models_dict, scaler, Xi, model_choice)
            probs_funding.append(p * 100)
            labels_x.append(format_currency(new_f, CURR))

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=labels_x, y=probs_funding, mode="lines+markers",
                                    line=dict(color="#2A9D8F", width=3),
                                    marker=dict(size=10)))
        fig_f.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Decision Threshold")
        fig_f.update_layout(title="Success Probability vs Funding Amount",
                             xaxis_title=f"Total Funding ({CURR})",
                             yaxis_title="Success Probability (%)",
                             yaxis=dict(range=[0, 100]),
                             height=380)
        st.plotly_chart(fig_f, width="stretch")

        # What-if slider
        new_fund_local = st.slider(f"Try a funding amount ({CURR})",
                                    int(convert_from_usd(10_000, CURR)),
                                    int(convert_from_usd(100_000_000, CURR)),
                                    int(base_funding))
        new_fund_usd = convert_to_usd(new_fund_local, CURR)
        Xi_new = build_X(new_fund_usd, base_rounds, base_age, base_category, base_country)
        _, new_prob_f = predict_single(models_dict, scaler, Xi_new, model_choice)
        direction = f"increased to {format_currency(new_fund_usd, CURR)}" if new_fund_usd > base_funding_usd else f"decreased to {format_currency(new_fund_usd, CURR)}"
        col_a, col_b = st.columns(2)
        col_a.metric("New Probability", f"{new_prob_f*100:.1f}%", f"{(new_prob_f-base_prob)*100:+.1f}%")
        with col_b:
            with st.spinner("Generating what-if insight..."):
                wi_text = get_whatif_explanation(base_prob, new_prob_f, "Total Funding", direction, CURR)
            render_text_box(wi_text)

    with tab2:
        rounds_range = list(range(0, 16))
        probs_rounds = []
        for r in rounds_range:
            Xi = build_X(base_funding_usd, r, base_age, base_category, base_country)
            _, p = predict_single(models_dict, scaler, Xi, model_choice)
            probs_rounds.append(p * 100)

        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=rounds_range, y=probs_rounds,
                                marker_color=["#E63946" if p < 50 else "#2A9D8F" for p in probs_rounds]))
        fig_r.add_hline(y=50, line_dash="dash", line_color="black")
        fig_r.update_layout(title="Success Probability vs Funding Rounds",
                             xaxis_title="Number of Funding Rounds",
                             yaxis_title="Success Probability (%)",
                             yaxis=dict(range=[0, 100]), height=380)
        st.plotly_chart(fig_r, width="stretch")

    with tab3:
        countries_compare = ["PAK", "USA", "GBR", "IND", "ARE", "DEU", "SGP", "CAN", "AUS", "FRA"]
        probs_country = []
        for c in countries_compare:
            Xi = build_X(base_funding_usd, base_rounds, base_age, base_category, c)
            _, p = predict_single(models_dict, scaler, Xi, model_choice)
            probs_country.append(p * 100)

        country_labels = {
            "PAK": "Pakistan", "USA": "USA", "GBR": "UK",
            "IND": "India", "ARE": "UAE", "DEU": "Germany",
            "SGP": "Singapore", "CAN": "Canada", "AUS": "Australia", "FRA": "France"
        }
        fig_c = go.Figure(go.Bar(
            x=[country_labels.get(c, c) for c in countries_compare],
            y=probs_country,
            marker_color=["#01411C" if c == "PAK" else ("#2A9D8F" if p >= 50 else "#E63946")
                           for c, p in zip(countries_compare, probs_country)],
        ))
        fig_c.add_hline(y=50, line_dash="dash", line_color="black")
        fig_c.update_layout(title="Success Probability by Country (same startup profile)",
                             yaxis_title="Success Probability (%)",
                             yaxis=dict(range=[0, 100]), height=380)
        st.plotly_chart(fig_c, width="stretch")
        st.caption("Pakistan data has been specifically incorporated into training, so predictions include local ecosystem signals.")


# ════════════════════════════════════════════════════
#  PAGE: STARTUP CHALLENGE
# ════════════════════════════════════════════════════
elif page == "Startup Challenge":
    st.markdown("""<div class="main-header">
      <h1>Startup Challenge</h1>
      <p>Play a founder scenario, allocate resources, and review model-backed coaching feedback</p>
    </div>""", unsafe_allow_html=True)

    sym = CURRENCY_SYMBOLS[CURR]

    SCENARIOS = [
        {
            "id": 1, "title": "The Bootstrapped Tech Startup",
            "desc": "You just got your first seed funding. Allocate resources to maximise your chances.",
            "budget_usd": 50_000, "industry": "Software", "country": "PAK",
            "hint": "In Pakistan's tech scene, product-market fit is crucial early on."
        },
        {
            "id": 2, "title": "The FinTech Challenger",
            "desc": "You have raised Series A. Compete in the crowded digital payments space.",
            "budget_usd": 2_000_000, "industry": "FinTech", "country": "PAK",
            "hint": "FinTech success in Pakistan requires regulatory compliance and trust-building."
        },
        {
            "id": 3, "title": "The E-Commerce Disruptor",
            "desc": "You are entering the B2C e-commerce market with strong backing.",
            "budget_usd": 5_000_000, "industry": "E-Commerce", "country": "PAK",
            "hint": "Logistics infrastructure is the make-or-break factor for Pakistani e-commerce."
        },
        {
            "id": 4, "title": "The Global EdTech Expansion",
            "desc": "Your edtech startup is scaling beyond Pakistan into the MENA region.",
            "budget_usd": 10_000_000, "industry": "EdTech", "country": "ARE",
            "hint": "Content localisation and partnerships are key for MENA market entry."
        },
    ]

    st.markdown('<div class="game-card"><h3>Founder Challenge</h3>'
                '<p style="color:#a8b2c1">Choose a scenario and allocate your budget strategically. '
                'The AI coach evaluates your choices and gives feedback after each round.</p></div>',
                unsafe_allow_html=True)

    col_score1, col_score2, col_score3 = st.columns(3)
    col_score1.metric("Score", st.session_state.game_score)
    col_score2.metric("Rounds Played", st.session_state.game_round)
    col_score3.metric("Best Outcome", "Success" if st.session_state.game_score > 0 else "—")

    st.markdown("---")

    scenario_titles = [f"Scenario {s['id']}: {s['title']}" for s in SCENARIOS]
    chosen_idx = st.selectbox("Choose Your Scenario", range(len(SCENARIOS)),
                               format_func=lambda i: scenario_titles[i])
    scenario = SCENARIOS[chosen_idx]

    budget_display = format_currency(scenario["budget_usd"], CURR)

    st.markdown(f"""
    <div class="game-card">
      <h3>{scenario['title']}</h3>
      <p style="color:#e0e0e0">{scenario['desc']}</p>
      <p><strong>Budget: {budget_display}</strong> &nbsp; | &nbsp; 
         Industry: {scenario['industry']} &nbsp; | &nbsp; 
         Country: {scenario['country']}</p>
      <p style="color:#d6e4ff">Analyst note: {scenario['hint']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Allocate Your Budget (%)")
    st.caption("Sliders must sum to 100%")

    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        pct_product = st.slider("Product Development", 0, 80, 35)
        pct_marketing = st.slider("Marketing & Sales", 0, 80, 25)
        pct_operations = st.slider("Operations", 0, 50, 15)
    with col_sl2:
        pct_team = st.slider("Team / HR", 0, 60, 20)
        pct_reserve = 100 - pct_product - pct_marketing - pct_operations - pct_team
        st.metric("Cash Reserve (auto)", f"{pct_reserve}%",
                   delta="Good buffer" if pct_reserve >= 5 else "Too low!")

    total_pct = pct_product + pct_marketing + pct_operations + pct_team + pct_reserve
    if total_pct != 100:
        st.warning(f"Allocation sums to {total_pct}%. Adjust sliders to reach 100%.")

    play_btn = st.button("Play Round", type="primary",
                          width="stretch", disabled=(total_pct != 100))

    if play_btn and total_pct == 100:
        with st.spinner("Simulating outcome..."):
            # Estimate funding from budget allocation
            rounds_est = 1 + pct_product // 20
            Xi = np.array([[
                np.log1p(scenario["budget_usd"]), rounds_est,
                np.log1p(2), np.log1p(365), 365, 2022,
                np.log1p(scenario["budget_usd"] / max(rounds_est, 1)), 0, 0,
                le_cat.transform([scenario["industry"]])[0]
                    if scenario["industry"] in le_cat.classes_ else 0,
                int(scenario["country"] == "PAK")
            ]])
            label, prob = predict_single(models_dict, scaler, Xi, model_choice)

            allocation = {
                "product": pct_product, "marketing": pct_marketing,
                "operations": pct_operations, "team": pct_team, "reserve": pct_reserve,
                "budget_display": budget_display,
            }
            feedback = get_gamified_feedback(
                scenario["title"], allocation, label, prob, CURR
            )

        # Score update
        points = int(prob * 100)
        st.session_state.game_score += points
        st.session_state.game_round += 1

        if label == 1:
            st.success(f"**Outcome: Success.** The startup is predicted to succeed with {prob*100:.1f}% confidence. +{points} points.")
        else:
            st.error(f"**Outcome: High risk.** Success probability is {prob*100:.1f}%. +{points} points.")

        # Allocation chart
        fig_pie = go.Figure(go.Pie(
            labels=["Product Dev", "Marketing", "Operations", "Team/HR", "Reserve"],
            values=[pct_product, pct_marketing, pct_operations, pct_team, pct_reserve],
            hole=0.4,
            marker_colors=["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653"]
        ))
        fig_pie.update_layout(title=f"Budget Allocation - {budget_display}", height=320)
        st.plotly_chart(fig_pie, width="stretch")

        st.markdown("**AI Coach Feedback:**")
        render_text_box(feedback)

        if st.button("Play Another Round"):
            st.rerun()


# ════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("""<div class="main-header">
      <h1>Model Performance</h1>
      <p>Detailed evaluation metrics for all three ML models</p>
    </div>""", unsafe_allow_html=True)

    tab_overview, tab_roc, tab_cm, tab_overfit = st.tabs([
        "Metrics Overview", "ROC-AUC Comparison", "Confusion Matrices", "Overfitting Check"
    ])

    with tab_overview:
        metric_rows = []
        for name, r in results.items():
            metric_rows.append({
                "Model": name,
                "Train Accuracy": f"{r.get('train_accuracy', r['accuracy'])*100:.2f}%",
                "Accuracy": f"{r['accuracy']*100:.2f}%",
                "Balanced Accuracy": f"{r['balanced_accuracy']*100:.2f}%",
                "Train-Test Gap": f"{r.get('overfit_gap', 0)*100:.2f}%",
                "Precision": f"{r['precision']*100:.2f}%",
                "Success Recall": f"{r['recall']*100:.2f}%",
                "Failure Recall": f"{r['failure_recall']*100:.2f}%",
                "F1-Score": f"{r['f1']*100:.2f}%",
                "ROC-AUC": f"{r['roc_auc']:.4f}",
                "PR-AUC": f"{r['pr_auc']:.4f}",
                "MCC": f"{r['mcc']:.4f}",
                "Threshold": f"{r['threshold']:.2f}",
            })
        df_metrics = pd.DataFrame(metric_rows)
        st.dataframe(df_metrics, width="stretch", hide_index=True)
        st.warning(
            "Plain accuracy can look high because most records are labeled as success. "
            "Use Balanced Accuracy, Failure Recall, ROC-AUC, and MCC to judge whether the model is truly useful."
        )

        # Bar chart comparison
        model_names = list(results.keys())
        metrics_to_plot = ["balanced_accuracy", "accuracy", "precision", "recall", "failure_recall", "f1", "roc_auc"]
        metric_labels = ["Balanced Accuracy", "Accuracy", "Precision", "Success Recall", "Failure Recall", "F1-Score", "ROC-AUC"]
        colors_map = {"Logistic Regression": "#264653", "Random Forest": "#2A9D8F", "XGBoost": "#E9C46A"}

        fig_bar = go.Figure()
        for mname in model_names:
            fig_bar.add_trace(go.Bar(
                name=mname,
                x=metric_labels,
                y=[results[mname][m] for m in metrics_to_plot],
                marker_color=colors_map[mname],
            ))
        fig_bar.update_layout(barmode="group", title="All Models - Metric Comparison",
                               yaxis=dict(range=[0, 1.05]), height=420)
        st.plotly_chart(fig_bar, width="stretch")

    with tab_roc:
        st.markdown("""
        **ROC-AUC** measures a model's ability to distinguish between successful and failed startups.
        - AUC = 1.0 → Perfect classifier
        - AUC = 0.5 → Random guessing
        """)
        auc_vals = {name: results[name]["roc_auc"] for name in results}
        fig_auc = go.Figure(go.Bar(
            x=list(auc_vals.keys()),
            y=list(auc_vals.values()),
            marker_color=["#264653", "#2A9D8F", "#E9C46A"],
            text=[f"{v:.4f}" for v in auc_vals.values()],
            textposition="outside"
        ))
        fig_auc.update_layout(title="ROC-AUC Score by Model",
                               yaxis=dict(range=[0.5, 1.0]),
                               height=380)
        st.plotly_chart(fig_auc, width="stretch")
        st.caption("Because the dataset is imbalanced, use balanced accuracy and failure recall with ROC-AUC before trusting plain accuracy.")

        pr_vals = {name: results[name]["pr_auc"] for name in results}
        fig_pr = go.Figure(go.Bar(
            x=list(pr_vals.keys()),
            y=list(pr_vals.values()),
            marker_color=["#264653", "#2A9D8F", "#E9C46A"],
            text=[f"{v:.4f}" for v in pr_vals.values()],
            textposition="outside"
        ))
        fig_pr.update_layout(title="Precision-Recall AUC by Model",
                             yaxis=dict(range=[0.5, 1.0]),
                             height=380)
        st.plotly_chart(fig_pr, width="stretch")

    with tab_cm:
        for mname, r in results.items():
            st.markdown(f"#### {mname}")
            cm = r["cm"]
            fig_cm, ax = plt.subplots(figsize=(4, 3.5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                           display_labels=["Failure", "Success"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{mname} — Confusion Matrix")
            st.pyplot(fig_cm, width="content")
            plt.close()

    with tab_overfit:
        overfit_rows = []
        for name, r in results.items():
            gap = float(r.get("overfit_gap", 0))
            risk = "Low" if gap < 0.03 else ("Moderate" if gap < 0.07 else "High")
            overfit_rows.append({
                "Model": name,
                "Training Accuracy": f"{r.get('train_accuracy', r['accuracy'])*100:.2f}%",
                "Test Accuracy": f"{r['accuracy']*100:.2f}%",
                "Gap": f"{gap*100:.2f}%",
                "Overfitting Risk": risk,
            })
        st.dataframe(pd.DataFrame(overfit_rows), width="stretch", hide_index=True)

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=[row["Model"] for row in overfit_rows],
            y=[float(row["Gap"].replace("%", "")) for row in overfit_rows],
            marker_color=["#22C55E" if row["Overfitting Risk"] == "Low" else "#EAB308" for row in overfit_rows],
            text=[row["Gap"] for row in overfit_rows],
            textposition="outside",
        ))
        fig_gap.update_layout(
            title="Train-Test Accuracy Gap",
            yaxis_title="Gap in percentage points",
            height=360,
        )
        st.plotly_chart(fig_gap, width="stretch")
        st.caption("A small train-test gap suggests the model is generalizing instead of memorizing the training data.")

    st.markdown("---")
    st.info("""
    **Training Notes:**
    - Dataset: Uploaded Crunchbase records merged with an expanded Pakistan startup ecosystem augmentation
    - Pakistan share is held near 25% of the combined training database
    - The model tunes its classification threshold on a validation split using balanced accuracy
    - Failure recall/specificity is reported because plain accuracy is misleading when most labels are success
    - Overfitting is checked by comparing training accuracy with held-out test accuracy
    """)


# ════════════════════════════════════════════════════
#  PAGE: PAKISTAN STARTUPS
# ════════════════════════════════════════════════════
elif page == "Pakistan Startups":
    st.markdown("""<div class="main-header">
      <h1>Pakistan Startup Ecosystem</h1>
      <p>Updated Pakistan startup data and ecosystem signals used alongside the global Crunchbase dataset</p>
    </div>""", unsafe_allow_html=True)

    pak_df = get_pakistan_df().copy()
    pak_model_df = df_all[df_all["country_code"].fillna("").str.upper() == "PAK"].copy()
    pak_model_df["funding_total_usd"] = pd.to_numeric(pak_model_df["funding_total_usd"], errors="coerce").fillna(0)
    pak_model_df["primary_sector"] = (
        pak_model_df["category_list"].fillna("Other").astype(str).str.split("|").str[0].replace("", "Other")
    )
    pak_model_df["founded_year"] = pd.to_datetime(pak_model_df["founded_at"], errors="coerce").dt.year
    pak_model_df["founded_year"] = pak_model_df["founded_year"].fillna(pak_model_df["founded_year"].median()).astype(int)

    pak_df["funding_display"] = pak_df["funding_total_usd"].apply(
        lambda x: format_currency(x, CURR)
    )
    pak_df["status_label"] = pak_df["status"].map({
        "operating": "Operating", "acquired": "Acquired", "closed": "Closed"
    })
    pak_df["last_funding_year"] = pd.to_datetime(pak_df["last_funding_at"], errors="coerce").dt.year.fillna(0).astype(int)

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    total_funding_usd = pak_df["funding_total_usd"].sum()
    c1.metric("Verified Pakistan Records", len(pak_df))
    c2.metric("Pakistan Model Rows", f"{len(pak_model_df):,}")
    c3.metric(f"Verified Funding ({CURR})", format_currency(total_funding_usd, CURR))
    c4.metric("Recent 2024-2025 Records", len(pak_df[pak_df["last_funding_year"] >= 2024]))

    st.markdown("---")

    # Funding chart: top verified startups by disclosed funding.
    pak_sorted = pak_df.sort_values("funding_total_usd", ascending=True).tail(22)
    fig_pak = go.Figure(go.Bar(
        y=pak_sorted["name"],
        x=pak_sorted["funding_total_usd"].apply(lambda x: convert_from_usd(x, CURR)),
        orientation="h",
        marker_color=["#E63946" if s == "closed" else ("#2A9D8F" if s == "operating" else "#264653")
                       for s in pak_sorted["status"]],
        text=pak_sorted["funding_total_usd"].apply(lambda x: format_currency(x, CURR)),
        textposition="outside",
    ))
    fig_pak.update_layout(
        title=f"Top Verified Pakistan Startups by Disclosed Funding ({CURR})",
        xaxis_title=f"Funding ({CURR})",
        height=660,
        margin=dict(l=160)
    )
    st.plotly_chart(fig_pak, width="stretch")

    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        st.subheader("Model Dataset by Sector")
        sector_counts = pak_model_df["primary_sector"].value_counts().head(10).sort_values()
        fig_sector = go.Figure(go.Bar(
            y=sector_counts.index,
            x=sector_counts.values,
            orientation="h",
            marker_color="#22D3EE",
            text=sector_counts.values,
            textposition="outside",
        ))
        fig_sector.update_layout(
            title="Pakistan Startup Sectors in Training Data",
            xaxis_title="Startup records",
            height=420,
            margin=dict(l=130),
        )
        st.plotly_chart(fig_sector, width="stretch")

    with chart_col2:
        st.subheader("Model Dataset by City")
        city_counts = pak_model_df["city"].fillna("Unknown").value_counts().head(8)
        fig_city = px.pie(
            values=city_counts.values,
            names=city_counts.index,
            title="Pakistan Startup Records by City",
            hole=0.48,
            color_discrete_sequence=["#7C3AED", "#06B6D4", "#22C55E", "#EAB308", "#F97316", "#EC4899"],
        )
        fig_city.update_layout(height=420)
        st.plotly_chart(fig_city, width="stretch")

    trend_df = pak_model_df[
        (pak_model_df["founded_year"] >= 2006) & (pak_model_df["founded_year"] <= 2025)
    ]["founded_year"].value_counts().sort_index()
    fig_trend = go.Figure(go.Scatter(
        x=trend_df.index,
        y=trend_df.values,
        mode="lines+markers",
        line=dict(color="#22C55E", width=3),
        marker=dict(size=8, color="#EAB308"),
        fill="tozeroy",
    ))
    fig_trend.update_layout(
        title="Pakistan Startup Formation Trend in the Model Dataset",
        xaxis_title="Founding year",
        yaxis_title="Startup records",
        height=380,
    )
    st.plotly_chart(fig_trend, width="stretch")

    st.subheader("Verified Pakistan Startup Directory")
    display_cols = ["name", "category_list", "city", "status_label", "funding_rounds", "funding_display"]
    display_df = pak_df.sort_values(["last_funding_year", "funding_total_usd"], ascending=[False, False])[display_cols].rename(columns={
        "name": "Startup", "category_list": "Industry",
        "city": "City", "status_label": "Status",
        "funding_rounds": "Rounds", "funding_display": f"Funding ({CURR})"
    })
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.info(f"""
    **Data Notes:**
    - Verified Pakistan records were expanded with newer ecosystem names reported in 2024-2025
    - Large training charts use the Pakistan portion of the combined model dataset, not only the verified directory
    - Some early-stage funding values are approximate disclosed estimates where exact ticket sizes were not public
    - Funding shown in **{CURRENCY_NAMES[CURR]}** (converted from USD at rate 1 USD = {EXCHANGE_RATES[CURR]:,.2f} {CURR})
    - Status values: Operating, Acquired/Exit, Closed
    - These records seed the Pakistan augmentation merged with the uploaded global Crunchbase dataset for model training
    """)
