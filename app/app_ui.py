# Styled UI prototype — run with: streamlit run app/app_ui.py
# The original bare-bones app remains in app/app.py as a fallback.

import html
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from validation import validate_inputs

# --- Load model + thresholds ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "model_RandomForest.pkl"))
thresholds = joblib.load(os.path.join(MODEL_DIR, "tier_thresholds.pkl"))
q1, q3 = thresholds["q1"], thresholds["q3"]

REGIONS = {
    "Select Region": None,
    "African Region (AFR)": "AFR",
    "Region of the Americas (AMR)": "AMR",
    "Eastern Mediterranean Region (EMR)": "EMR",
    "European Region (EUR)": "EUR",
    "South-East Asia Region (SEAR)": "SEAR",
    "Western Pacific Region (WPR)": "WPR",
}

DEFAULTS = {
    "region_label": "Select Region",
    "gdp": 3200.0,
    "pop_density": 145.0,
    "urban_pct": 52.0,
    "sanitation_pct": 86.0,
    "rainfall": 1800.0,
    "temp": 26.5,
}

TIER_COLORS = {
    "Low": "#27ae60",
    "Medium": "#e8912d",
    "High": "#e74c3c",
}

HELP = {
    "header": (
        "Enter country-level indicators, then click Predict. The tool estimates "
        "median malaria cases and assigns a Low, Medium, or High tier."
    ),
    "region": (
        "WHO epidemiological region for the scenario. Region is a categorical "
        "predictor that captures broad geographic patterns in the training data."
    ),
    "gdp": (
        "Median GDP per capita in US dollars. Higher income can reflect better "
        "health infrastructure that influences reported malaria burden."
    ),
    "pop_density": (
        "Median population density (people per km²). Density relates to urbanization "
        "and contact patterns that can affect transmission."
    ),
    "urban_pct": (
        "Median share of the population living in urban areas (%). Urbanization "
        "can change exposure to mosquito habitats and health services."
    ),
    "sanitation_pct": (
        "Median share of the population with access to sanitation (%). Sanitation "
        "is a WASH indicator linked to overall public health conditions."
    ),
    "rainfall": (
        "Median annual rainfall in millimeters. Rainfall influences mosquito "
        "breeding sites and seasonal transmission potential."
    ),
    "temp": (
        "Median temperature in degrees Celsius. Temperature affects mosquito "
        "survival and malaria transmission dynamics."
    ),
    "predicted_count": (
        "Model output: estimated median malaria case count for the inputs you "
        "provided, using the saved Random Forest pipeline."
    ),
    "classification": (
        "Tier label from training quartiles: below Q1 is Low, above Q3 is High, "
        "otherwise Medium."
    ),
    "predict_btn": "Validate inputs, run the model, and update the results panel.",
    "reset_btn": "Clear results and restore default example values in all fields.",
}


def label_with_tip(title: str, tip: str) -> str:
    safe_title = html.escape(title)
    safe_tip = html.escape(tip)
    return (
        f'<p class="card-label">{safe_title}'
        f'<span class="info-tip" tabindex="0" role="button" aria-label="Help: {safe_title}">'
        f'<span class="info-icon" aria-hidden="true">i</span>'
        f'<span class="tip-text">{safe_tip}</span></span></p>'
    )


def result_label_with_tip(title: str, tip: str) -> str:
    safe_title = html.escape(title)
    safe_tip = html.escape(tip)
    return (
        f'<span class="result-label">{safe_title}'
        f'<span class="info-tip" tabindex="0" role="button" aria-label="Help: {safe_title}">'
        f'<span class="info-icon" aria-hidden="true">i</span>'
        f'<span class="tip-text">{safe_tip}</span></span></span>'
    )


st.set_page_config(
    page_title="Malaria Case Prediction Tool",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}


.stApp {
    background: linear-gradient(rgba(12, 45, 82, 0.72), rgba(12, 45, 82, 0.72)),
        url('https://images.pexels.com/photos/4189472/pexels-photo-4189472.jpeg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
}

.header-bar {
    background: #000;
    border-radius: 14px;
    padding: 1.1rem 1.5rem;
    text-align: center;
    margin-bottom: 1.25rem;
}
.header-bar h1 {
    color: #fff;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.01em;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    justify-content: center;
}
.header-subtitle {
    color: #d7e9f8;
    font-size: 0.9rem;
    margin: 0.45rem 0 0;
    line-height: 1.35;
}
.a11y-hint {
    color: #b8d4ea;
    font-size: 0.78rem;
    margin-top: 0.35rem;
}

.info-tip {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-left: 0.35rem;
    vertical-align: middle;
    cursor: help;
    outline: none;
}
.info-icon {
    width: 1.05rem;
    height: 1.05rem;
    border-radius: 999px;
    background: #7eb8da;
    color: #0d2a45;
    font-size: 0.72rem;
    font-weight: 800;
    line-height: 1.05rem;
    text-align: center;
    display: inline-block;
}
.tip-text {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    bottom: calc(100% + 8px);
    width: 240px;
    background: #0d2a45;
    color: #fff;
    font-size: 0.78rem;
    font-weight: 500;
    line-height: 1.35;
    padding: 0.55rem 0.65rem;
    border-radius: 8px;
    text-align: left;
    z-index: 1000;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    pointer-events: none;
    transition: opacity 0.15s ease;
}
.info-tip:hover .tip-text,
.info-tip:focus .tip-text,
.info-tip:focus-visible .tip-text {
    visibility: visible;
    opacity: 1;
}
.result-label .info-tip .tip-text {
    left: 0;
    transform: none;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] * {
    font-size: 0.9rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(26, 58, 92, 0.92) !important;
    border-color: transparent !important;
    border-radius: 16px !important;
    padding: 0.35rem 0.15rem 0.15rem !important;
    margin-bottom: 0.75rem;
}

.card-label {
    color: #fff;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 0 0 0.15rem 0.35rem;
}
div[data-testid="stNumberInput"] input {
    background-color: #7eb8da !important;
    color: #0d2a45 !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600;
    min-height: 42px;
}
div[data-testid="stSelectbox"] * {
    background-color: #7eb8da !important;
    color: #0d2a45 !important;
    -webkit-text-fill-color: #0d2a45 !important;
    border-color: transparent !important;
    font-weight: 600;
}
div[data-testid="stSelectbox"] * {
    border-radius: 999px !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 42px;
    overflow: hidden;
}
div[data-testid="stSelectbox"] svg {
    fill: #0d2a45 !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    box-shadow: 0 0 0 2px rgba(126, 184, 218, 0.45) !important;
}
div[data-testid="stNumberInput"] > div > div {
    background: transparent !important;
}
div[data-testid="stNumberInput"] button,
div[data-testid="stNumberInput"] button * {
    color: #ffffff !important;
    fill: #ffffff !important;
    background: transparent !important;
}
.results-panel {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 1rem;
    min-height: 118px;
    padding: 0.35rem 0.5rem;
}
.result-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.result-label {
    color: #fff;
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.25;
}
.result-pill {
    background: #7eb8da;
    color: #0d2a45;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.55rem 1.25rem;
    min-width: 150px;
    text-align: center;
    white-space: nowrap;
}
.tier-pill {
    color: #fff;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.55rem 1.25rem;
    min-width: 150px;
    text-align: center;
}

.stButton button,
div[data-testid="stButton"] button  {
    background: #00bcd4 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 1rem !important;
}
.stButton button:hover,
div[data-testid="stButton"] button:hover {
    background: #00acc1 !important;
    color: #fff !important;
    border: none !important;
}
.validation-stack {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin: 0.25rem 0 1rem 0;
}
.validation-banner {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.4;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #0b2d3d;
}
.validation-banner.validation-warning {
    background: rgba(160, 210, 156, 0.72);
    border-left: 5px solid #3b7c44;
}
.validation-banner.validation-error {
    background: rgba(229, 171, 171, 0.75);
    border-left: 5px solid #a93a3a;
}
.validation-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.4rem;
    height: 1.4rem;
    border-radius: 50%;
    background: rgba(255,255,255,0.35);
}
</style>
""",
    unsafe_allow_html=True,
)

if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.tier = None

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

if "validation_errors" not in st.session_state:
    st.session_state.validation_errors = []

if "validation_warnings" not in st.session_state:
    st.session_state.validation_warnings = []

def reset_form():
    st.session_state.prediction = None
    st.session_state.tier = None
    st.session_state.validation_errors = []
    st.session_state.validation_warnings = []
    st.session_state.reset_counter += 1

def render_validation_messages():
    combined_messages = [
        (message, "error") for message in st.session_state.validation_errors
    ] + [
        (message, "warning") for message in st.session_state.validation_warnings
    ]

    if not combined_messages:
        return

    banner_html = "".join(
        (
            "<div class=\"validation-banner validation-{kind}\">"
            "<span class=\"validation-icon\">{icon}</span>"
            "<span>{message}</span>"
            "</div>"
        ).format(
            kind=kind,
            icon="⚠️" if kind == "error" else "ℹ️",
            message=html.escape(message),
        )
        for message, kind in combined_messages
    )

    st.markdown(
        f"""
<div class="validation-stack">{banner_html}</div>
""",
        unsafe_allow_html=True,
    )


header_tip = html.escape(HELP["header"])
st.markdown(
    f"""
<div class="header-bar">
  <h1>Malaria Case Prediction Tool
    <span class="info-tip" tabindex="0" role="button" aria-label="About this tool">
      <span class="info-icon" aria-hidden="true">i</span>
      <span class="tip-text">{header_tip}</span>
    </span>
  </h1>
  <p class="header-subtitle">This tool is a calculator using Country-level scenario inputs to estimate median malaria cases (also on a country level).</p>
  <p class="a11y-hint">Hover or focus the <strong>i</strong> icons for brief field descriptions.</p>
</div>
""",
    unsafe_allow_html=True,
)

form_key = st.session_state.reset_counter

row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3 = st.columns(3)
row3_col1, row3_col2 = st.columns([1, 2])

with row1_col1:
    with st.container(border=True):
        st.markdown(
            label_with_tip("WHO Epidemiological Region", HELP["region"]),
            unsafe_allow_html=True,
        )
        region_label = st.selectbox(
            "WHO Epidemiological Region",
            list(REGIONS.keys()),
            index=list(REGIONS.keys()).index(DEFAULTS["region_label"]),
            label_visibility="collapsed",
            help=HELP["region"],
            key=f"region_{form_key}",
        )

with row1_col2:
    with st.container(border=True):
        st.markdown(label_with_tip("GDP (USD)", HELP["gdp"]), unsafe_allow_html=True)
        gdp = st.number_input(
            "GDP (USD)",
            value=DEFAULTS["gdp"],
            min_value=0.0,
            step=100.0,
            label_visibility="collapsed",
            help=HELP["gdp"],
            key=f"gdp_{form_key}",
        )

with row1_col3:
    with st.container(border=True):
        st.markdown(
            label_with_tip("Population Density (p/km²)", HELP["pop_density"]),
            unsafe_allow_html=True,
        )
        pop_density = st.number_input(
            "Population Density (p/km²)",
            value=DEFAULTS["pop_density"],
            min_value=0.0,
            step=1.0,
            label_visibility="collapsed",
            help=HELP["pop_density"],
            key=f"pop_density_{form_key}",
        )

with row2_col1:
    with st.container(border=True):
        st.markdown(
            label_with_tip("Urban population (%)", HELP["urban_pct"]),
            unsafe_allow_html=True,
        )
        urban_pct = st.number_input(
            "Urban population (%)",
            value=DEFAULTS["urban_pct"],
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            label_visibility="collapsed",
            help=HELP["urban_pct"],
            key=f"urban_pct_{form_key}",
        )

with row2_col2:
    with st.container(border=True):
        st.markdown(
            label_with_tip("Sanitation access (%)", HELP["sanitation_pct"]),
            unsafe_allow_html=True,
        )
        sanitation_pct = st.number_input(
            "Sanitation access (%)",
            value=DEFAULTS["sanitation_pct"],
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            label_visibility="collapsed",
            help=HELP["sanitation_pct"],
            key=f"sanitation_pct_{form_key}",
        )

with row2_col3:
    with st.container(border=True):
        st.markdown(label_with_tip("Rainfall (mm)", HELP["rainfall"]), unsafe_allow_html=True)
        rainfall = st.number_input(
            "Rainfall (mm)",
            value=DEFAULTS["rainfall"],
            min_value=0.0,
            step=10.0,
            label_visibility="collapsed",
            help=HELP["rainfall"],
            key=f"rainfall_{form_key}",
        )

with row3_col1:
    with st.container(border=True):
        st.markdown(label_with_tip("Temperature (C)", HELP["temp"]), unsafe_allow_html=True)
        temp = st.number_input(
            "Temperature (C)",
            value=DEFAULTS["temp"],
            min_value=-50.0,
            max_value=60.0,
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
            help=HELP["temp"],
            key=f"temp_{form_key}",
        )

with row3_col2:
    count_display = (
        f"{st.session_state.prediction:,.4f} Cases"
        if st.session_state.prediction is not None
        else "— Cases"
    )
    tier_display = st.session_state.tier if st.session_state.tier else "—"
    tier_color = TIER_COLORS.get(st.session_state.tier, "#7eb8da")

    with st.container(border=True):
        st.markdown(
            f"""
<div class="results-panel">
    <div class="result-row">
        {result_label_with_tip("Predicted Case Count", HELP["predicted_count"])}
        <span class="result-pill">{count_display}</span>
    </div>
    <div class="result-row">
        {result_label_with_tip("Case Count Classification", HELP["classification"])}
        <span class="tier-pill" style="background:{tier_color};">{tier_display}</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
with btn_col2:
    predict_clicked = st.button(
        "Predict",
        use_container_width=True,
        help=HELP["predict_btn"],
    )
with btn_col3:
    reset_clicked = st.button(
        "Reset",
        use_container_width=True,
        help=HELP["reset_btn"],
    )

if reset_clicked:
    reset_form()
    st.rerun()

if predict_clicked:
    region_code = REGIONS[region_label]
    errors, warnings = validate_inputs(
        region_code, gdp, pop_density, urban_pct, sanitation_pct, rainfall, temp
    )
    st.session_state.validation_errors = errors
    st.session_state.validation_warnings = warnings

    if errors:
        st.session_state.prediction = None
        st.session_state.tier = None
    else:
        input_df = pd.DataFrame(
            [
                {
                    "Region_Code": region_code,
                    "Median_GDP": gdp,
                    "Median_Population_Density": pop_density,
                    "Median_Urban_pct": urban_pct,
                    "Median_Sanitation_Access_pct": sanitation_pct,
                    "Median_Rainfall": rainfall,
                    "Median_Temp": temp,
                }
            ]
        )

        prediction = np.expm1(model.predict(input_df)[0])

        if prediction < q1:
            tier = "Low"
        elif prediction > q3:
            tier = "High"
        else:
            tier = "Medium"

        st.session_state.prediction = prediction
        st.session_state.tier = tier

    st.rerun()

render_validation_messages()