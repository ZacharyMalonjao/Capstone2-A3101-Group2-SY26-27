"""Input validation for the malaria prediction Streamlit apps."""

VALID_REGION_CODES = {"AFR", "AMR", "EMR", "EUR", "SEAR", "WPR"}

# Observed ranges in ipynb_notebooks/final_extract_dataset.csv (n=99 countries).
TRAINING_RANGES = {
    "gdp": (248.34, 34324.67),
    "pop_density": (3.22, 1266.91),
    "urban_pct": (14.05, 91.90),
    "sanitation_pct": (8.85, 99.80),
    "rainfall": (31.10, 3739.20),
    "temp": (4.39, 30.35),
}

FIELD_LABELS = {
    "gdp": "GDP (USD)",
    "pop_density": "Population Density (p/km²)",
    "urban_pct": "Urban population (%)",
    "sanitation_pct": "Sanitation access (%)",
    "rainfall": "Rainfall (mm)",
    "temp": "Temperature (C)",
}


def _is_finite_number(value):
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


def validate_inputs(region_code, gdp, pop_density, urban_pct, sanitation_pct, rainfall, temp):
    """Return (errors, warnings) for Streamlit form values."""
    errors = []
    warnings = []

    if not region_code:
        errors.append("Please select a WHO epidemiological region.")
    elif region_code not in VALID_REGION_CODES:
        errors.append("Region code is invalid. Choose a supported WHO region.")

    numeric_fields = {
        "gdp": gdp,
        "pop_density": pop_density,
        "urban_pct": urban_pct,
        "sanitation_pct": sanitation_pct,
        "rainfall": rainfall,
        "temp": temp,
    }

    for field, value in numeric_fields.items():
        label = FIELD_LABELS[field]

        if not _is_finite_number(value):
            errors.append(f"{label} must be a valid number.")
            continue

        if field in {"urban_pct", "sanitation_pct"}:
            if value < 0 or value > 100:
                errors.append(f"{label} must be between 0 and 100.")
                continue
        elif value < 0:
            errors.append(f"{label} cannot be negative.")
            continue

        if field == "temp" and (value < -50 or value > 60):
            errors.append(f"{label} must be between -50 and 60 °C.")
            continue

        train_min, train_max = TRAINING_RANGES[field]
        if value < train_min or value > train_max:
            warnings.append(
                f"{label} ({value:,.2f}) is outside the training data range "
                f"({train_min:,.2f}–{train_max:,.2f}). Predictions may be less reliable."
            )

    return errors, warnings
