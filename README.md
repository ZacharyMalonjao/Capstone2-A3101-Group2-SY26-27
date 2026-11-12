# Estimating Relative Malaria Case Count and Case Count Classification Across Nations Using Regression-Based Modeling

A regression-based machine learning project examining **99 countries** to estimate relative national malaria case burden using climatic, socioeconomic, and demographic indicators, and classify countries into Low, Medium, or High case-count tiers.

The project combines **Python data preparation, statistical analysis, and machine learning** with a Streamlit web application to make country-level malaria estimates more interpretable and accessible.

## Key Findings

* **Random Forest** achieved the best cross-validated performance, with a mean R² of **0.5956** on the log scale.
* The Random Forest model achieved **90% tier classification accuracy**, correctly classifying 18 out of 20 held-out countries.
* Random Forest correctly identified **all 6 High-burden countries** in the test set.
* **Sanitation access** emerged as the most consistently influential factor across all five regression models and was statistically significant in OLS regression.
* European countries had significantly lower malaria case counts than the African reference region.
* Rainfall, temperature, GDP per capita, population density, and urbanization were not statistically significant predictors in the OLS analysis.
* Adding total population as a sensitivity-analysis predictor improved Random Forest's cross-validated R² from **0.596 to 0.649**.
* The results suggest that reporting predictions through **Low, Medium, and High tiers** is more appropriate than relying on exact case-count estimates, given the substantial prediction errors.

These findings are promising but should be interpreted cautiously because of the relatively small dataset and the variability in model performance.

## Application

The final Streamlit application allows users to enter country-level demographic, socioeconomic, and climatic indicators to estimate relative malaria case burden.

The application provides:

* Estimated national malaria case count
* Low, Medium, or High case-count classification
* Color-coded prediction results
* Plain-language explanations of each classification
* Input validation and range warnings

**Streamlit Application:**

[Launch the Malaria Prediction Tool](https://malaria-prediction-tool.streamlit.app/)

**UI Mockup (Canva):**

[View the application UI mockup](https://canva.link/rp7bbex0mnm60k5)

> **Note:** The application is hosted on Streamlit Community Cloud, which may sleep after inactivity. If a wake-up screen appears, click to restart the application.

## Project Questions

The analysis was organized around one main research question and four supporting questions.

### 1. What factors are associated with national malaria case burden?

Which climatic, socioeconomic, and demographic indicators are significantly associated with malaria case counts across countries?

### 2. How accurately can malaria case counts be estimated?

How well can regression-based models estimate national malaria case burden using the selected indicators?

### 3. Which regression model performs best?

How do the candidate models compare in terms of R², RMSE, and MAE?

### 4. How accurately can countries be classified?

How effectively can the selected model classify countries into Low, Medium, and High malaria case-count tiers?

### 5. Which factors influence the predictions?

Which indicators contribute most significantly to the estimation of national malaria case burden?

---

# Data

### Dataset

**Relative Malaria Case Count Prediction Dataset**

* **Records collected:** 106 countries
* **Final records:** 99 countries
* **Period:** 2015–2024
* **Observations:** One median value per country
* **Predictors:** 7
* **Model terms:** 11 after one-hot encoding of WHO region
* **Target variable:** Median_Malaria_Cases
* **Derived output:** Case_Count_Tier (Low / Medium / High)

**Data Sources:**

* **WHO Global Health Observatory:** Malaria case data
* **World Bank:** GDP per capita, population density, urban population percentage, sanitation access, and total population
* **Open-Meteo:** Rainfall and temperature data
* **REST Countries API:** Capital-city coordinates used to retrieve climate data

The dataset combines public health, socioeconomic, demographic, and climatic indicators to examine their relationships with national malaria case burden.

The target variable represents the **median annual malaria case count per country from 2015 to 2024**, rather than malaria incidence or a time-series forecast.

### Limitations

The dataset has several limitations that should be considered when interpreting the results:

* The relatively small sample of 99 countries makes model evaluation sensitive to the countries included in each validation fold.
* Seven countries were excluded because of missing study variables, potentially limiting the generalizability of the findings.
* Climate measurements were obtained using capital-city coordinates, which may not accurately represent malaria-endemic areas within each country.
* The analysis uses a cross-sectional country-level representation and cannot capture changes in malaria burden over time.
* Approximately 40% of the variance in log-transformed case counts remains unexplained by the seven predictors.

---

# Data Preparation

The raw datasets from multiple public sources required cleaning, aggregation, and standardization before they could be used for statistical analysis and machine learning.

Python and Pandas were used to prepare the country-level dataset.

### Data Quality Checks

The preparation process included:

* Checking data completeness across countries
* Identifying missing observations
* Validating country-level records
* Performing Shapiro-Wilk normality tests on yearly country-level series
* Conducting Variance Inflation Factor (VIF) analysis to screen for multicollinearity
* Reviewing predictor distributions and correlations

### Missing Values

Countries with missing values in any study variable or WHO malaria data were excluded rather than imputed.

Of the 106 countries initially collected, 99 were retained for the final analysis.

This approach avoided introducing imputed values into the country-level dataset, although it reduced the final sample size.

### Aggregation

The original data contained annual observations from 2015 to 2024.

These observations were consolidated into one record per country using median aggregation.

Climate variables required additional processing:

* Temperature was averaged from daily observations into annual means.
* Rainfall was summed into yearly totals.
* Annual country-level values were subsequently aggregated into their respective 10-year median values.

### Feature Engineering

The following transformations and derived variables were prepared:

* Median aggregation of annual observations
* `log1p` transformation of the malaria case-count target
* One-hot encoding of WHO region
* Standardization of numerical predictors
* Quartile-based classification of malaria case counts into Low, Medium, and High tiers

### Encoding and Data Leakage Prevention

WHO region was treated as a categorical variable, while the remaining predictors were treated as continuous numerical variables.

One-hot encoding and standardization were incorporated into the machine learning pipelines.

The Q1 and Q3 tier thresholds were calculated using the raw training-set malaria case counts only.

Encoding, scaling, and
