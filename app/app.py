import os

import joblib
import pandas as pd
import streamlit as st

from validation import validate_inputs

# --- Load model + thresholds ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "model_ElasticNet.pkl"))
thresholds = joblib.load(os.path.join(MODEL_DIR, "tier_thresholds.pkl"))
q1, q3 = thresholds["q1"], thresholds["q3"]

st.title("Malaria Case Count Predictor (Bare Bones Test)")

# --- Inputs, no styling, just functional ---
region = st.selectbox("Region Code", ["AFR", "AMR", "EMR", "EUR", "SEAR", "WPR"])
gdp = st.number_input("Median GDP", value=0.0, min_value=0.0)
pop_density = st.number_input("Median Population Density", value=0.0, min_value=0.0)
urban_pct = st.number_input("Median Urban %", value=0.0, min_value=0.0, max_value=100.0)
sanitation_pct = st.number_input(
    "Median Sanitation Access %", value=0.0, min_value=0.0, max_value=100.0
)
rainfall = st.number_input("Median Rainfall", value=0.0, min_value=0.0)
temp = st.number_input("Median Temp", value=0.0, min_value=-50.0, max_value=60.0)

if st.button("Predict"):
    errors, warnings = validate_inputs(
        region, gdp, pop_density, urban_pct, sanitation_pct, rainfall, temp
    )

    for message in errors:
        st.error(message)
    for message in warnings:
        st.warning(message)

    if not errors:
        input_df = pd.DataFrame(
            [
                {
                    "Region_Code": region,
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

        st.write(f"**Predicted Median Malaria Cases:** {prediction:,.0f}")
        st.write(f"**Case Count Tier:** {tier}")
