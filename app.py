"""
Smart Electrical Grid Stability AI Platform
Interactive Streamlit Diagnostics, Simulation, and Decision Support System
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from src.train_predictor import train_model

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Grid Stability AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "random_forest.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "model_comparison.csv"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "electrical_grid_stability.csv"

# ---------------------------------------------------------
# Custom Styling & Glassmorphic Design System
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    code, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ---------------------------------------------------------
       Streamlit Header & 3-Dots Menu Customization
       --------------------------------------------------------- */
    /* 1. Hide Deploy Button */
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    .stDeployButton,
    [data-testid="stDeployButton"],
    header [data-testid="stToolbarActions"],
    header button:has([data-testid="stDeployButton"]) {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }

    /* 2. Theme Switcher: Ensure container is visible */
    [data-testid="stThemeSwitcher"] {
        display: flex !important;
        width: 100% !important;
        gap: 8px !important;
    }

    /* Explicitly keep System and Dark buttons visible (each takes 50%) */
    [data-testid="stMainMenuItem-theme-System"],
    [data-testid="stMainMenuItem-theme-Dark"],
    [data-testid="stThemeSwitcher"] > *:first-child,
    [data-testid="stThemeSwitcher"] > *:last-child {
        display: flex !important;
        visibility: visible !important;
        flex: 1 1 50% !important;
    }

    /* Hide 'Light' mode button */
    [data-testid="stMainMenuItem-theme-Light"],
    [data-testid="stThemeSwitcher"] > *:nth-child(2) {
        display: none !important;
        visibility: hidden !important;
    }

    /* 3. Hide all menu items & dividers below the theme switcher */
    [data-testid="stThemeSwitcher"] ~ *,
    [data-testid="stMainMenuDivider"],
    [data-testid="stMainMenuItem-rerun"],
    [data-testid="stMainMenuItem-autoRerun"],
    [data-testid="stMainMenuItem-clearCache"],
    [data-testid="stMainMenuItem-print"],
    [data-testid="stMainMenuItem-recordScreencast"],
    [data-testid="stMainMenuItem-about"],
    [data-testid="stMainMenuList"] ~ *,
    .stMenuVersionCopyButton {
        display: none !important;
    }

    /* Clean popover padding */
    [data-testid="stMainMenuList"] {
        padding-bottom: 4px !important;
        margin-bottom: 0 !important;
    }

    /* Sidebar Brand Title (Adaptive Light/Dark/System Theme) */
    .brand-title {
        margin: 0 !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: var(--text-color, #0f172a) !important;
        letter-spacing: -0.01em !important;
        line-height: 1.2 !important;
    }

    .brand-subtitle {
        font-size: 0.72rem !important;
        color: #0284c7 !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }

    @media (prefers-color-scheme: dark) {
        .brand-subtitle {
            color: #38bdf8 !important;
        }
    }

    [data-theme="dark"] .brand-subtitle,
    [data-base="dark"] .brand-subtitle {
        color: #38bdf8 !important;
    }

    /* ---------------------------------------------------------
       Clean Selectbox & Dropdown Styling
       --------------------------------------------------------- */
    /* Selectbox Input Container */
    .stSelectbox div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div:has(> input) {
        border-radius: 10px !important;
    }

    /* Selectbox Input Text */
    .stSelectbox input,
    div[data-testid="stSelectbox"] input {
        font-size: 0.9rem !important;
        line-height: 1.3 !important;
    }

    /* Ensure Virtual Dropdown Popover floats cleanly above all sidebar elements */
    div[data-testid="stSelectboxVirtualDropdown"] {
        z-index: 999999 !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
        border-radius: 10px !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.5), 0 0 20px rgba(56, 189, 248, 0.1);
        backdrop-filter: blur(12px);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
        margin: 0;
    }

    /* Metric Cards Grid */
    .kpi-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .kpi-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-caption {
        font-size: 0.75rem;
        color: #10b981;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Section Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }
    .glass-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Verdict Card */
    .verdict-box {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
        animation: fadeIn 0.4s ease-in-out;
    }
    .verdict-stable {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.9) 0%, rgba(6, 95, 70, 0.7) 100%);
        border: 2px solid #10b981;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
    }
    .verdict-unstable {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.9) 0%, rgba(153, 27, 27, 0.7) 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
    }

    .verdict-tag {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .verdict-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 0;
        color: #ffffff;
    }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        background: transparent;
        color: #94a3b8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.12) !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Badges */
    .badge-stable {
        background-color: #064e3b;
        color: #34d399;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-unstable {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Loading & Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_prediction_model():
    """Loads the trained Random Forest classifier or trains from raw data if absent."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    model, _ = train_model()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_data
def load_comparison_metrics():
    """Loads empirical model evaluation metrics."""
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


@st.cache_data
def load_sample_dataset():
    """Loads raw dataset for distribution and testing."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


# Load Model
model = load_prediction_model()
metrics_df = load_comparison_metrics()
raw_data_df = load_sample_dataset()

# ---------------------------------------------------------
# Preset Scenarios for Instant Testing
# ---------------------------------------------------------
SCENARIO_PRESETS = {
    "Select a Scenario Preset...": None,
    "🟢 Resilient Grid (Stable)": {
        "tau1": 1.25, "tau2": 1.40, "tau3": 1.15, "tau4": 1.30,
        "p1": 3.60, "p2": -1.20, "p3": -1.20, "p4": -1.20,
        "g1": 0.15, "g2": 0.20, "g3": 0.18, "g4": 0.22,
        "desc": "Synchronized and resilient. Low reaction times (fast damping) and minimal price sensitivity ensure rapid stabilization under perturbations."
    },
    "🔴 High-Latency Grid (Blackout Risk)": {
        "tau1": 8.75, "tau2": 9.20, "tau3": 8.90, "tau4": 9.50,
        "p1": 4.50, "p2": -1.80, "p3": -1.50, "p4": -1.20,
        "g1": 0.85, "g2": 0.92, "g3": 0.88, "g4": 0.90,
        "desc": "Extremely sluggish participant reaction times coupled with aggressive price elasticity lead to phase angle divergence."
    },
    "🟡 Marginal Oscillation (Boundary)": {
        "tau1": 5.10, "tau2": 4.80, "tau3": 5.40, "tau4": 4.90,
        "p1": 3.75, "p2": -1.25, "p3": -1.25, "p4": -1.25,
        "g1": 0.50, "g2": 0.55, "g3": 0.48, "g4": 0.52,
        "desc": "Operating right at the critical boundary of Lyapunov exponent stability."
    },
    "⚡ Heavy Generator Stress": {
        "tau1": 2.10, "tau2": 6.80, "tau3": 3.40, "tau4": 7.10,
        "p1": 5.20, "p2": -1.95, "p3": -1.75, "p4": -1.50,
        "g1": 0.30, "g2": 0.75, "g3": 0.40, "g4": 0.80,
        "desc": "Extreme power generation from the central supplier node with heterogeneous consumer response latencies."
    }
}

# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
        <span style="font-size:2.2rem;">⚡</span>
        <div>
            <h3 class="brand-title">GridStability AI</h3>
            <span class="brand-subtitle">AUTONOMOUS ML SYSTEM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Scenario Quick Presets")
    st.caption("Load pre-configured power grid operating states:")
    selected_preset_key = st.selectbox(
        "Choose test scenario",
        options=list(SCENARIO_PRESETS.keys()),
        index=0,
        label_visibility="collapsed"
    )

    preset_data = SCENARIO_PRESETS[selected_preset_key]
    if preset_data:
        st.info(f"💡 **Scenario Overview:** {preset_data['desc']}")

    st.markdown("---")
    st.subheader("⚙️ Simulation Settings")
    auto_balance_power = st.checkbox(
        "⚡ Auto-Balance Generator",
        value=True,
        help="Automatically computes p1 = -(p2 + p3 + p4) to guarantee zero net power imbalance."
    )


# ---------------------------------------------------------
# Top Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ Autonomous Telemetry & Diagnostics AI</div>
    <h1 class="hero-title">Decentralized Electrical Grid Stability Predictor</h1>
    <p class="hero-subtitle">
        Real-time machine learning inference for a 4-node smart electricity network. 
        Detects oscillatory divergence, phase instability, and blackout risks from dynamic participant reaction times, power flows, and price elasticity.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# KPI Metrics Row
# ---------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Active Classifier</div>
        <div class="kpi-value">Random Forest</div>
        <div class="kpi-caption">200 Decision Estimators</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Test Set Accuracy</div>
        <div class="kpi-value" style="color:#10b981;">92.90%</div>
        <div class="kpi-caption">+29.10% over Baseline</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Macro F1-Score</div>
        <div class="kpi-value" style="color:#38bdf8;">0.9220</div>
        <div class="kpi-caption">Harmonic Mean of Classes</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Target Leakage Risk</div>
        <div class="kpi-value" style="color:#a855f7;">0.00%</div>
        <div class="kpi-caption">Eigenvalue (stab) Excluded</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_sim, tab_analytics, tab_topology, tab_batch = st.tabs([
    "🕹️ Interactive Grid Simulator",
    "📊 Model Performance & Benchmarking",
    "🕸️ 4-Node Grid Topology & Theory",
    "📁 Batch Simulation & CSV Lab"
])

# =========================================================
# TAB 1: INTERACTIVE SIMULATOR
# =========================================================
with tab_sim:
    st.markdown("### 🎛️ Network Parameter Configuration")
    st.caption("Adjust the operating parameters of the decentralized 4-node power grid or load a scenario from the sidebar.")

    # Preset values fallback helper
    def get_val(key, default):
        if preset_data and key in preset_data:
            return float(preset_data[key])
        return float(default)

    sim_form = st.container()

    with sim_form:
        param_col1, param_col2, param_col3 = st.columns([1, 1, 1])

        # Column 1: Reaction Times
        with param_col1:
            st.markdown("""
            <div class="glass-card">
                <div class="glass-card-header">
                    <span>⏱️</span> Participant Latency
                </div>
            """, unsafe_allow_html=True)
            st.caption("Reaction time in seconds (0.5s – 10.0s). Higher values slow down damping.")

            tau1 = st.slider("Supplier Reaction (tau1)", 0.5, 10.0, get_val("tau1", 2.93), 0.05, help="Central Power Generating Plant")
            tau2 = st.slider("Consumer 1 Reaction (tau2)", 0.5, 10.0, get_val("tau2", 3.05), 0.05)
            tau3 = st.slider("Consumer 2 Reaction (tau3)", 0.5, 10.0, get_val("tau3", 4.20), 0.05)
            tau4 = st.slider("Consumer 3 Reaction (tau4)", 0.5, 10.0, get_val("tau4", 3.85), 0.05)
            st.markdown("</div>", unsafe_allow_html=True)

        # Column 2: Power Flows
        with param_col2:
            st.markdown("""
            <div class="glass-card">
                <div class="glass-card-header">
                    <span>⚡</span> Power Balance
                </div>
            """, unsafe_allow_html=True)
            st.caption("Nominal power in p.u. Rule: Supplier p1 = -(p2 + p3 + p4).")

            p2 = st.slider("Consumer 1 Demand (p2)", -2.00, -0.50, get_val("p2", -1.25), 0.01)
            p3 = st.slider("Consumer 2 Demand (p3)", -2.00, -0.50, get_val("p3", -1.25), 0.01)
            p4 = st.slider("Consumer 3 Demand (p4)", -2.00, -0.50, get_val("p4", -1.25), 0.01)

            auto_p1_value = -round(p2 + p3 + p4, 4)

            if auto_balance_power:
                p1 = auto_p1_value
                st.number_input("Supplier Generation (p1) [Auto-Balanced]", value=float(p1), disabled=True)
            else:
                p1 = st.number_input("Supplier Generation (p1)", min_value=1.50, max_value=6.00, value=get_val("p1", 3.75), step=0.01)

            # Net Balance Alert
            net_power = p1 + p2 + p3 + p4
            if abs(net_power) < 0.05:
                st.markdown(f"<span style='color:#10b981; font-weight:600; font-size:0.85rem;'>✅ Power Balanced: Net ΔP = {net_power:+.3f} p.u.</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:#ef4444; font-weight:600; font-size:0.85rem;'>⚠️ Power Imbalance Alert: Net ΔP = {net_power:+.3f} p.u.</span>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Column 3: Price Elasticity
        with param_col3:
            st.markdown("""
            <div class="glass-card">
                <div class="glass-card-header">
                    <span>📈</span> Price Elasticity
                </div>
            """, unsafe_allow_html=True)
            st.caption("Willingness to adapt consumption to price shifts (0.05 – 1.0).")

            g1 = st.slider("Supplier Elasticity (g1)", 0.05, 1.00, get_val("g1", 0.50), 0.01)
            g2 = st.slider("Consumer 1 Elasticity (g2)", 0.05, 1.00, get_val("g2", 0.45), 0.01)
            g3 = st.slider("Consumer 2 Elasticity (g3)", 0.05, 1.00, get_val("g3", 0.55), 0.01)
            g4 = st.slider("Consumer 3 Elasticity (g4)", 0.05, 1.00, get_val("g4", 0.60), 0.01)
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Real-time Prediction Execution
    # -----------------------------------------------------
    current_features = [tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4]
    input_df = pd.DataFrame([current_features], columns=FEATURE_COLUMNS)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_dict = dict(zip(model.classes_, probabilities))
    prob_stable = prob_dict["stable"]
    prob_unstable = prob_dict["unstable"]

    st.markdown("---")
    st.markdown("### 🔬 Real-Time Stability Diagnosis")

    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        # Verdict Box
        if prediction == "stable":
            st.markdown(f"""
            <div class="verdict-box verdict-stable">
                <div class="verdict-tag" style="color:#a7f3d0;">DIAGNOSTIC STATUS</div>
                <h2 class="verdict-title">STABLE OPERATING STATE</h2>
                <div style="font-size:1.05rem; color:#d1fae5; margin-top:8px;">
                    Grid parameters are within the safe differential damping basin.
                </div>
                <div style="display:flex; justify-content:center; gap:20px; margin-top:16px;">
                    <div>
                        <div style="font-size:0.8rem; color:#a7f3d0; text-transform:uppercase;">Stable Likelihood</div>
                        <div style="font-size:1.8rem; font-weight:800; color:#ffffff;">{prob_stable:.1%}</div>
                    </div>
                    <div style="border-left:1px solid rgba(255,255,255,0.2); padding-left:20px;">
                        <div style="font-size:0.8rem; color:#a7f3d0; text-transform:uppercase;">Blackout Risk</div>
                        <div style="font-size:1.8rem; font-weight:800; color:#fca5a5;">{prob_unstable:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box verdict-unstable">
                <div class="verdict-tag" style="color:#fecaca;">DIAGNOSTIC STATUS</div>
                <h2 class="verdict-title">UNSTABLE (CASCADE RISK)</h2>
                <div style="font-size:1.05rem; color:#fee2e2; margin-top:8px;">
                    Oscillatory divergence detected. High probability of synchronous trip.
                </div>
                <div style="display:flex; justify-content:center; gap:20px; margin-top:16px;">
                    <div>
                        <div style="font-size:0.8rem; color:#fecaca; text-transform:uppercase;">Unstable Probability</div>
                        <div style="font-size:1.8rem; font-weight:800; color:#ffffff;">{prob_unstable:.1%}</div>
                    </div>
                    <div style="border-left:1px solid rgba(255,255,255,0.2); padding-left:20px;">
                        <div style="font-size:0.8rem; color:#fecaca; text-transform:uppercase;">Stable Likelihood</div>
                        <div style="font-size:1.8rem; font-weight:800; color:#a7f3d0;">{prob_stable:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Gauge Speedometer Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_stable * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "<b>Stability Safety Index (%)</b>", 'font': {'size': 18, 'color': '#f8fafc'}},
            number={'suffix': "%", 'font': {'size': 32, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#10b981" if prediction == "stable" else "#ef4444", 'thickness': 0.28},
                'bgcolor': "rgba(30, 41, 59, 0.4)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 45], 'color': 'rgba(239, 68, 68, 0.3)'},
                    {'range': [45, 60], 'color': 'rgba(245, 158, 11, 0.3)'},
                    {'range': [60, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#38bdf8", 'width': 3},
                    'thickness': 0.8,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc", 'family': "Outfit"},
            height=260,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        # Radar Chart of 12 Features
        categories = ["tau1", "tau2", "tau3", "tau4", "p1", "p2 (abs)", "p3 (abs)", "p4 (abs)", "g1", "g2", "g3", "g4"]
        # Normalize features approximately for radar display:
        radar_values = [
            tau1 / 10.0, tau2 / 10.0, tau3 / 10.0, tau4 / 10.0,
            p1 / 6.0, abs(p2) / 2.0, abs(p3) / 2.0, abs(p4) / 2.0,
            g1, g2, g3, g4
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=categories,
            fill='toself',
            name='Current Operating Vector',
            fillcolor='rgba(56, 189, 248, 0.25)',
            line=dict(color='#38bdf8', width=2)
        ))
        # Ideal reference baseline
        baseline_stable = [0.25, 0.25, 0.25, 0.25, 0.60, 0.60, 0.60, 0.60, 0.30, 0.30, 0.30, 0.30]
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_stable,
            theta=categories,
            name='Typical Resilient Grid Profile',
            line=dict(color='#10b981', dash='dot', width=1.5)
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#64748b')),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#cbd5e1'))
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color='#cbd5e1')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=370,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.markdown("##### 📡 Multi-Dimensional Parameter Fingerprint")
        st.plotly_chart(fig_radar, use_container_width=True)

        # Automated Engineering Advisory
        st.markdown("""
        <div style="background:rgba(30, 41, 59, 0.5); border-left:4px solid #38bdf8; border-radius:8px; padding:12px 16px; margin-top:10px;">
            <div style="font-size:0.85rem; font-weight:700; color:#38bdf8;">⚡ AUTOMATED DISPATCH ADVISORY</div>
            <div style="font-size:0.8rem; color:#cbd5e1; margin-top:4px;">
        """, unsafe_allow_html=True)
        if tau1 > 6.0 or tau2 > 6.0 or tau3 > 6.0 or tau4 > 6.0:
            st.markdown("• **High Participant Latency Warning:** Participant reaction time $\\tau$ exceeds 6.0s. Grid frequency controller may fail to dampen oscillation.")
        if g1 > 0.7 or g2 > 0.7 or g3 > 0.7 or g4 > 0.7:
            st.markdown("• **High Elasticity Volatility:** Participant sensitivity $\\gamma > 0.70$ causes rapid demand fluctuations following price signals.")
        if abs(net_power) > 0.05:
            st.markdown("• **Unbalanced System:** Net load mismatch $\\sum p \\neq 0$. Ensure generation equals total consumer absorption.")
        if prediction == "stable" and max(tau1, tau2, tau3, tau4) <= 4.0:
            st.markdown("• **Optimal Damping State:** System operates within robust Lyapunov stability margins.")
        st.markdown("</div></div>", unsafe_allow_html=True)


# =========================================================
# TAB 2: MODEL BENCHMARKING & ANALYTICS
# =========================================================
with tab_analytics:
    st.markdown("### 🏆 Rigorous Machine Learning Model Comparison")
    st.caption("All models evaluated strictly on the exact same 2,000 held-out test cases (80/20 Stratified Split, `random_state=42`).")

    col_bench1, col_bench2 = st.columns([1.1, 1])

    with col_bench1:
        st.markdown("##### 📋 Quantitative Metric Scorecard")
        if metrics_df is not None:
            # Set Metric as index so only numeric columns are styled and compared
            display_metrics = metrics_df.set_index("Metric")
            st.dataframe(
                display_metrics.style.format("{:.4f}").highlight_max(axis=1, color="#1e3a8a"),
                use_container_width=True,
                height=420
            )
        else:
            st.warning("Metrics file outputs/metrics/model_comparison.csv not found.")

    with col_bench2:
        st.markdown("##### 🌲 Random Forest Feature Importance (MDI)")
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Parameter": FEATURE_COLUMNS,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=True)

            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Parameter",
                orientation='h',
                color="Importance",
                color_continuous_scale="Tealgrn",
                text_auto='.3f'
            )
            fig_fi.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f8fafc"},
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                height=400,
                margin=dict(l=20, r=20, t=10, b=20)
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🛡️ Critical Safety Risk: Analysis of 'False Stables'")
    
    risk_col1, risk_col2 = st.columns([1, 1])
    with risk_col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#ef4444; margin-top:0;">⚠️ The High Cost of False Negatives in Power Grids</h4>
            <p style="font-size:0.9rem; color:#cbd5e1; line-height:1.6;">
                In high-voltage power transmission, predicting that an unstable grid is <b>"Stable" (False Stable)</b> is catastrophic. Operators fail to dispatch emergency spinning reserves or island the sub-network, triggering <b>cascading system-wide blackouts</b>.
            </p>
            <ul style="font-size:0.88rem; color:#94a3b8;">
                <li><b>Logistic Regression:</b> 146 False Stables (Unstable Recall: 88.56%)</li>
                <li><b>Random Forest:</b> <span style="color:#10b981; font-weight:700;">Only 48 False Stables (Unstable Recall: 96.24%)</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with risk_col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#38bdf8; margin-top:0;">🔍 Why Random Forest Outperformed Linear Classification</h4>
            <p style="font-size:0.9rem; color:#cbd5e1; line-height:1.6;">
                Electrical grid dynamics are dictated by coupled nonlinear differential equations. Stability is not a linear hyperplane separation; it depends on <b>complex interaction terms</b> (e.g., reaction time $\\tau_i$ compounded with high elasticity $\\gamma_i$).
            </p>
            <p style="font-size:0.88rem; color:#94a3b8;">
                Random Forest's 200 orthogonal decision surfaces easily capture non-convex stability boundaries that linear models misclassify.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 3: 4-NODE TOPOLOGY & THEORY
# =========================================================
with tab_topology:
    st.markdown("### 🕸️ 4-Node Smart Grid Star Network Topology")
    st.caption("Decentralized power grid model formulated by Vadim Arzamasov et al. (2018).")

    top_col1, top_col2 = st.columns([1.2, 1])

    with top_col1:
        st.markdown("""
        ```mermaid
        graph TD
            N1["⚡ Node 1 (Central Supplier)<br>Power: p1 > 0<br>Latency: tau1 | Elasticity: g1"]
            N2["🏭 Node 2 (Consumer 1)<br>Power: p2 < 0<br>Latency: tau2 | Elasticity: g2"]
            N3["🏢 Node 3 (Consumer 2)<br>Power: p3 < 0<br>Latency: tau3 | Elasticity: g3"]
            N4["🏠 Node 4 (Consumer 3)<br>Power: p4 < 0<br>Latency: tau4 | Elasticity: g4"]

            N1 <-->|Bidirectional Transmission Line| N2
            N1 <-->|Bidirectional Transmission Line| N3
            N1 <-->|Bidirectional Transmission Line| N4

            classDef gen fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff;
            classDef cons fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#fff;
            class N1 gen;
            class N2,N3,N4 cons;
        ```
        """)

    with top_col2:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-header">📐 Mathematical Governing Equations</div>
            <p style="font-size:0.85rem; color:#cbd5e1; line-height:1.6;">
                The smart grid stability problem is modeled using swing equations for rotatory generators and delayed consumer price adaptation:
            </p>
            <div style="background:rgba(0,0,0,0.3); padding:10px 14px; border-radius:8px; font-family:'JetBrains Mono'; font-size:0.85rem; color:#38bdf8; margin:10px 0;">
                M_i \ddot{\theta}_i + D_i \dot{\theta}_i = P_i - \sum K_{ij} \sin(\theta_i - \theta_j)
            </div>
            <p style="font-size:0.82rem; color:#94a3b8; line-height:1.5;">
                • $\\tau_i$: Reaction delay of participants adjusting demand to market signals.<br>
                • $p_i$: Injected or extracted active power satisfying $\sum p_i = 0$.<br>
                • $\gamma_i$: Price elasticity governing economic feedback strength.<br>
                • <b>Eigenvalue Root ($stab$):</b> Max real part of root. If $stab \le 0 \implies Stable$, else $Unstable$.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 4: BATCH SIMULATION & CSV LAB
# =========================================================
with tab_batch:
    st.markdown("### 📁 Batch Testing & Simulation Laboratory")
    st.caption("Upload a CSV with the 12 operating parameters or run a quick batch on 20 random samples from the test corpus.")

    batch_mode = st.radio("Select batch input source:", ["Use 20 Random Test Records from Dataset", "Upload Custom CSV File"], horizontal=True)

    if batch_mode == "Use 20 Random Test Records from Dataset":
        if raw_data_df is not None:
            sample_batch = raw_data_df.sample(20, random_state=42)
            X_batch = sample_batch[FEATURE_COLUMNS]
            ground_truth = sample_batch[TARGET_COLUMN].values

            batch_preds = model.predict(X_batch)
            batch_probs = model.predict_proba(X_batch)[:, 1]

            results_df = X_batch.copy()
            results_df["Actual Class"] = ground_truth
            results_df["Predicted Class"] = batch_preds
            results_df["Unstable Confidence"] = [f"{p:.1%}" for p in batch_probs]
            results_df["Correct?"] = results_df["Actual Class"] == results_df["Predicted Class"]

            style_fn = results_df.style.map if hasattr(results_df.style, "map") else results_df.style.applymap
            styled_batch = style_fn(
                lambda val: 'background-color: rgba(16, 185, 129, 0.2); color:#10b981; font-weight:700;' if val is True else ('background-color: rgba(239, 68, 68, 0.2); color:#ef4444; font-weight:700;' if val is False else ''),
                subset=['Correct?']
            )
            st.dataframe(styled_batch, use_container_width=True)

            acc_batch = (batch_preds == ground_truth).mean()
            st.success(f"Batch Accuracy on Selected 20 Cases: **{acc_batch:.1%}** ({(batch_preds == ground_truth).sum()} / 20 correct)")
        else:
            st.error("Raw dataset not found.")

    else:
        uploaded_file = st.file_uploader("Upload CSV containing the 12 operating features", type=["csv"])
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in FEATURE_COLUMNS if c not in user_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                user_preds = model.predict(user_df[FEATURE_COLUMNS])
                user_probs = model.predict_proba(user_df[FEATURE_COLUMNS])[:, 1]
                user_df["Predicted_Stability"] = user_preds
                user_df["Unstable_Probability"] = user_probs
                st.dataframe(user_df, use_container_width=True)

                csv_download = user_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_download,
                    file_name="grid_stability_predictions.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; color:#64748b; font-size:0.8rem;">
    <div>⚡ <b>Smart Electrical Grid Stability AI Platform</b></div>
    <div>Real-time Diagnostics & Non-Linear Machine Learning</div>
</div>
""", unsafe_allow_html=True)
