# Styled UI prototype — run with: streamlit run app/app_ui.py
# The original bare-bones app remains in app/app.py as a fallback.

import os

import joblib
import pandas as pd
import streamlit as st

# --- Load model + thresholds ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "model_Lasso.pkl"))
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

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #7eb8da !important;
    color: #0d2a45 !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600;
    min-height: 42px;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    box-shadow: 0 0 0 2px rgba(126, 184, 218, 0.45) !important;
}
div[data-testid="stNumberInput"] > div > div,
div[data-testid="stSelectbox"] > div > div {
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

.stButton > button {
    background: #00bcd4 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 1rem !important;
}
.stButton > button:hover {
    background: #00acc1 !important;
    color: #fff !important;
    border: none !important;
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


def reset_form():
    st.session_state.prediction = None
    st.session_state.tier = None
    st.session_state.reset_counter += 1


st.markdown(
    '<div class="header-bar"><h1>Malaria Case Prediction Tool</h1></div>',
    unsafe_allow_html=True,
)

form_key = st.session_state.reset_counter

row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3 = st.columns(3)
row3_col1, row3_col2 = st.columns([1, 2])

with row1_col1:
    with st.container(border=True):
        st.markdown(
            '<p class="card-label">WHO Epidemiological Region</p>',
            unsafe_allow_html=True,
        )
        region_label = st.selectbox(
            "WHO Epidemiological Region",
            list(REGIONS.keys()),
            index=list(REGIONS.keys()).index(DEFAULTS["region_label"]),
            label_visibility="collapsed",
            key=f"region_{form_key}",
        )

with row1_col2:
    with st.container(border=True):
        st.markdown('<p class="card-label">GDP (USD)</p>', unsafe_allow_html=True)
        gdp = st.number_input(
            "GDP (USD)",
            value=DEFAULTS["gdp"],
            min_value=0.0,
            step=100.0,
            label_visibility="collapsed",
            key=f"gdp_{form_key}",
        )

with row1_col3:
    with st.container(border=True):
        st.markdown(
            '<p class="card-label">Population Density (p/km²)</p>',
            unsafe_allow_html=True,
        )
        pop_density = st.number_input(
            "Population Density (p/km²)",
            value=DEFAULTS["pop_density"],
            min_value=0.0,
            step=1.0,
            label_visibility="collapsed",
            key=f"pop_density_{form_key}",
        )

with row2_col1:
    with st.container(border=True):
        st.markdown(
            '<p class="card-label">Urban population (%)</p>',
            unsafe_allow_html=True,
        )
        urban_pct = st.number_input(
            "Urban population (%)",
            value=DEFAULTS["urban_pct"],
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            label_visibility="collapsed",
            key=f"urban_pct_{form_key}",
        )

with row2_col2:
    with st.container(border=True):
        st.markdown(
            '<p class="card-label">Sanitation access (%)</p>',
            unsafe_allow_html=True,
        )
        sanitation_pct = st.number_input(
            "Sanitation access (%)",
            value=DEFAULTS["sanitation_pct"],
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            label_visibility="collapsed",
            key=f"sanitation_pct_{form_key}",
        )

with row2_col3:
    with st.container(border=True):
        st.markdown('<p class="card-label">Rainfall (mm)</p>', unsafe_allow_html=True)
        rainfall = st.number_input(
            "Rainfall (mm)",
            value=DEFAULTS["rainfall"],
            min_value=0.0,
            step=10.0,
            label_visibility="collapsed",
            key=f"rainfall_{form_key}",
        )

with row3_col1:
    with st.container(border=True):
        st.markdown('<p class="card-label">Temperature (C)</p>', unsafe_allow_html=True)
        temp = st.number_input(
            "Temperature (C)",
            value=DEFAULTS["temp"],
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
            key=f"temp_{form_key}",
        )

with row3_col2:
    count_display = (
        f"{st.session_state.prediction:,.0f} Cases"
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
        <span class="result-label">Predicted Case Count</span>
        <span class="result-pill">{count_display}</span>
    </div>
    <div class="result-row">
        <span class="result-label">Case Count<br>Classification</span>
        <span class="tier-pill" style="background:{tier_color};">{tier_display}</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
with btn_col2:
    predict_clicked = st.button("Predict", use_container_width=True)
with btn_col3:
    reset_clicked = st.button("Reset", use_container_width=True)

if reset_clicked:
    reset_form()
    st.rerun()

if predict_clicked:
    region_code = REGIONS[region_label]
    if region_code is None:
        st.error("Please select a WHO epidemiological region before predicting.")
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

        prediction = model.predict(input_df)[0]

        if prediction < q1:
            tier = "Low"
        elif prediction > q3:
            tier = "High"
        else:
            tier = "Medium"

        st.session_state.prediction = prediction
        st.session_state.tier = tier
        st.rerun()
