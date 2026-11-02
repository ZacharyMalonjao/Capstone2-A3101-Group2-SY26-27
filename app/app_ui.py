# Styled UI prototype — run with: streamlit run app/app_ui.py
# The original bare-bones app remains in app/app.py as a fallback.

import html
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from validation import TRAINING_RANGES, validate_inputs

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

TIER_DESCRIPTIONS = {
    "Low": "Among the lowest quarter of countries in this study's dataset by estimated malaria case count.",
    "Medium": "In the middle half of countries in this study's dataset by estimated malaria case count.",
    "High": "Among the highest quarter of countries in this study's dataset by estimated malaria case count.",
}

TIER_NOTE = "Tiers compare countries within this study's dataset. They are not official WHO risk classifications."


def training_range_reference(field: str, unit: str) -> str:
    minimum, maximum = TRAINING_RANGES[field]
    return (
        f"Observed training-data range across 99 countries: "
        f"{minimum:,.2f}–{maximum:,.2f} {unit}. This is a reference range, "
        "not a recommended target."
    )


HELP = {
    "region": (
        "WHO epidemiological region for the country. This is a categorical "
        "predictor with no measurement unit or reporting period; choose the "
        "matching WHO region."
    ),
    "gdp": (
        "GDP per capita in current US dollars per person per year, reported "
        "annually. Enter the median of annual values from 2015–2024, not a "
        "monthly amount. Higher income can reflect health infrastructure that "
        "influences malaria burden. "
        + training_range_reference("gdp", "USD/person/year")
    ),
    "pop_density": (
        "Population density in people per km², reported annually. Enter the "
        "median of annual values from 2015–2024; this is a density, not a "
        "monthly or yearly count. Density relates to urbanization and contact "
        "patterns that can affect transmission. "
        + training_range_reference("pop_density", "people/km²")
    ),
    "urban_pct": (
        "Share of the population living in urban areas, as a percentage, "
        "reported annually. Enter the median of annual values from 2015–2024. "
        "Urbanization can change exposure to mosquito habitats and health "
        "services. "
        + training_range_reference("urban_pct", "%")
    ),
    "sanitation_pct": (
        "Percentage of the population using at least basic sanitation services "
        "(World Bank indicator SH.STA.BASS.ZS), reported annually. Enter the "
        "median of annual values from 2015–2024. "
        + training_range_reference("sanitation_pct", "%")
    ),
    "rainfall": (
        "Total rainfall accumulated over a full year, in millimeters per year "
        "(not a monthly amount). Enter the median of annual totals from "
        "2015–2024. Rainfall influences mosquito breeding sites and seasonal "
        "transmission potential. "
        + training_range_reference("rainfall", "mm/year")
    ),
    "temp": (
        "Annual mean temperature in degrees Celsius, calculated from daily "
        "temperatures. Enter the median of annual means from 2015–2024. "
        "Temperature affects mosquito survival and malaria transmission "
        "dynamics. "
        + training_range_reference("temp", "°C")
    ),
    "predict_btn": "Validate inputs, run the model, and update the results panel.",
    "reset_btn": "Clear results and restore default example values in all fields.",
}


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
    max-width: 1400px;
}

.stApp {
    background: #071523;
    font-family: 'Inter', sans-serif;
}

.header-bar {
    background: #030b13;
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
.header-bar {
    background: #030b13;
    border: 2px solid #ffffff;
    border-radius: 14px;
    padding: 1.1rem 1.5rem;
    text-align: center;
    margin-bottom: 1.25rem;
}
.header-subtitle {
    color: #e2edf7;
    font-size: 0.9rem;
    margin: 0.45rem 0 0;
    line-height: 1.35;
}
.national-level {
    color: #9bdcf0;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 0.4rem 0 0;
}
.result-scope {
    color: #9bdcf0;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 0;
    text-align: center;
}
.how-to {
    color: #d2e2ef;
    font-size: 0.9rem;
    margin: 0 0 0.9rem 0;
}

.a11y-hint {
    color: #d2e2ef;
    font-size: 0.78rem;
    margin-top: 0.35rem;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] * {
    font-size: 0.9rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #10283d !important;
    border-color: transparent !important;
    border-radius: 16px !important;
    padding: 0.35rem 0.15rem 0.15rem !important;
    margin-bottom: 0.75rem;
    overflow: visible !important;
}

div[data-testid="stNumberInput"] input {
    background-color: #e3edf4 !important;
    color: #10283d !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600;
    min-height: 42px;
}
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stSelectbox"] [data-baseweb="select"] * {
    background-color: #e3edf4 !important;
    color: #10283d !important;
    -webkit-text-fill-color: #10283d !important;
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
    fill: #10283d !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    box-shadow: 0 0 0 2px rgba(126, 184, 218, 0.45) !important;
}
div[data-testid="stNumberInput"] > div > div {
    background: transparent !important;
}
div[data-testid="stNumberInput"] button[aria-label="Decrement"],
div[data-testid="stNumberInput"] button[aria-label="Increment"] {
    color: #ffffff !important;
    background: transparent !important;
}
div[data-testid="stNumberInput"] button[aria-label="Decrement"] svg,
div[data-testid="stNumberInput"] button[aria-label="Increment"] svg {
    fill: #ffffff !important;
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
    background: #e3edf4;
    color: #10283d;
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

.tier-help {
    margin-top: 0.1rem;
}
.tier-desc {
    color: #e2edf7;
    font-size: 0.88rem;
    line-height: 1.4;
    margin: 0 0 0.35rem 0;
}
.tier-note {
    color: #d2e2ef;
    font-size: 0.76rem;
    font-style: italic;
    line-height: 1.35;
    margin: 0;
}

.stButton button,
div[data-testid="stButton"] button  {
    background: #006b78 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 1rem !important;
}
.stButton button:hover,
div[data-testid="stButton"] button:hover {
    background: #005762 !important;
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

div[data-testid="stTooltipContent"] {
    max-width: min(22rem, calc(100vw - 2rem)) !important;
    white-space: normal !important;
    overflow-wrap: anywhere;
    font-size: 0.85rem;
    line-height: 1.4;
}
div[data-baseweb="popover"] {
    max-width: calc(100vw - 1rem) !important;
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


st.markdown(
    """
<div class="header-bar">
  <h1>Malaria Case Prediction Tool</h1>
  <p class="national-level">National-level estimate</p>
  <p class="header-subtitle">This tool estimates malaria case counts at the country level, not for provinces or local areas.</p>
  <p class="a11y-hint">Use the help icon beside each input for field definitions and units.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="how-to"><b>How to use:</b> select a WHO region, enter the six '
    "country-level indicator values, click <b>Predict</b>, then read the "
    "estimate in the results panel.</p>",
    unsafe_allow_html=True,
)

with st.expander("About this tool", expanded=False):
    st.markdown(
        "Predictions are rough estimates, which is why results are also shown as "
        "tiers. They are based on median values from 2015–2024 for 99 countries, "
        "using data from WHO, the World Bank, and Open-Meteo. This tool is for "
        "research and education, not clinical or policy decisions."
    )

form_key = st.session_state.reset_counter

count_display = (
    f"{st.session_state.prediction:,.0f} Cases"
    if st.session_state.prediction is not None
    else "— Cases"
)
tier_display = st.session_state.tier if st.session_state.tier else "—"
tier_color = TIER_COLORS.get(st.session_state.tier, "#9bdcf0")
tier_help_html = (
    f'<div class="tier-help">'
    f'<p class="tier-desc">{html.escape(TIER_DESCRIPTIONS[st.session_state.tier])}</p>'
    f'<p class="tier-note">{html.escape(TIER_NOTE)}</p>'
    f'</div>'
    if st.session_state.tier
    else ""
)
empty_note_html = ""
if st.session_state.prediction is None:
    empty_msg = (
        "No prediction yet. Fix the inputs on the left and try again."
        if st.session_state.validation_errors
        else "Enter values and click Predict."
    )
    empty_note_html = f'<p class="tier-desc">{empty_msg}</p>'

left, right = st.columns([3, 2], gap="large")

# 1. INPUTS 
with left:
    with st.container(border=True):
        region_label = st.selectbox(
            "WHO Epidemiological Region",
            list(REGIONS.keys()),
            index=list(REGIONS.keys()).index(DEFAULTS["region_label"]),
            help=HELP["region"],
            key=f"region_{form_key}",
        )

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row1_col1:
        with st.container(border=True):
            gdp = st.number_input(
                "GDP (USD)",
                value=DEFAULTS["gdp"],
                min_value=0.0,
                step=100.0,
                help=HELP["gdp"],
                key=f"gdp_{form_key}",
            )

    with row1_col2:
        with st.container(border=True):
            pop_density = st.number_input(
                "Population Density (p/km²)",
                value=DEFAULTS["pop_density"],
                min_value=0.0,
                step=1.0,
                help=HELP["pop_density"],
                key=f"pop_density_{form_key}",
            )

    with row1_col3:
        with st.container(border=True):
            urban_pct = st.number_input(
                "Urban population (%)",
                value=DEFAULTS["urban_pct"],
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                help=HELP["urban_pct"],
                key=f"urban_pct_{form_key}",
            )

    with row2_col1:
        with st.container(border=True):
            sanitation_pct = st.number_input(
                "Sanitation access (%)",
                value=DEFAULTS["sanitation_pct"],
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                help=HELP["sanitation_pct"],
                key=f"sanitation_pct_{form_key}",
            )

    with row2_col2:
        with st.container(border=True):
            rainfall = st.number_input(
                "Rainfall (mm)",
                value=DEFAULTS["rainfall"],
                min_value=0.0,
                step=10.0,
                help=HELP["rainfall"],
                key=f"rainfall_{form_key}",
            )

    with row2_col3:
        with st.container(border=True):
            temp = st.number_input(
                "Temperature (C)",
                value=DEFAULTS["temp"],
                min_value=-50.0,
                max_value=60.0,
                step=0.1,
                format="%.1f",
                help=HELP["temp"],
                key=f"temp_{form_key}",
            )
   

# 2. ACTION + 3. RESULT
with right:
    predict_col, reset_col = st.columns(2)
    with predict_col:
        predict_clicked = st.button(
            "Predict",
            width="stretch",
            help=HELP["predict_btn"],
        )
    with reset_col:
        reset_clicked = st.button(
            "Reset",
            width="stretch",
            help=HELP["reset_btn"],
        )

    render_validation_messages()

    with st.container(border=True):
        st.markdown(
            f"""
<div class="results-panel">
    <p class="result-scope">National-level estimate</p>
    <div class="result-row">
        <span class="result-label">Predicted Case Count</span>
        <span class="result-pill">{count_display}</span>
    </div>
    <div class="result-row">
        <span class="result-label">Case Count Classification</span>
        <span class="tier-pill" style="background:{tier_color};">{tier_display}</span>
    </div>{tier_help_html}{empty_note_html}
</div>
""",
            unsafe_allow_html=True,
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