Estimating Relative Malaria Case Count and Case Count Classification Across Nations Using Regression-Based Modeling

Can a handful of public climate, socioeconomic, and demographic indicators estimate a country's malaria burden well enough to sort it into a Low / Medium / High tier? Short answer: yes, with caveats. A Random Forest placed 18 of 20 held-out countries (90%) in the correct tier, and sanitation access emerged as the most consistently influential factor across all five models.

Live app · UI mockup · Capstone 2, Mapúa Malayan Digital College (AY 2025–2026)

1. Project Title & Overview

Project name: Estimating Relative Malaria Case Count and Case Count Classification Across Nations Using Regression-Based Modeling

Description: Most malaria prediction studies focus on a single country, rely on hard-to-explain models, or examine climate, socioeconomic, and demographic factors in isolation. This project combines all three types of indicators for 99 countries and compares six models to estimate each country's median annual malaria case count. Each estimate is then converted into an easy-to-read Low, Medium, or High tier and delivered through a web application.

Objective and scope:

Estimate national malaria case burden (median of 2015–2024) from seven cross-sectional predictors.
Compare candidate regression models on R², RMSE, and MAE and select the best performer.
Classify predictions into quartile-based tiers and deliver them in a Streamlit prototype.
Scope is country-level and cross-sectional. It is not subnational, and it is not a time-series forecast.

Methods and technologies: Python, pandas, scikit-learn (MLR, Ridge, Lasso, Elastic Net, Random Forest), statsmodels (OLS inference), repeated 5×5 cross-validation, Streamlit.

2. Key Findings / Results
Result	Value
Best model	Random Forest (cross-validated mean R² 0.596 on the log scale; RMSE 2.82; MAE 2.22)
Tier classification accuracy	90% (18 of 20 test countries) vs. 50–60% for linear models and 45% for the baseline
High-burden countries caught	6 of 6 (Random Forest). The linear models found only 1–2 of 6
Most influential factor	Sanitation access (significant in OLS, ranked #1 by Random Forest importance, 0.59)
Significant predictors (OLS, α = .05)	Sanitation access and the European region. Climate, population density, GDP, and urbanization were not significant
OLS explanatory power	R² = .698, adjusted R² = .660, F(11, 87) = 18.27, p < .001
System testing	9 of 9 test cases passed on the final build
Usability testing (n = 3)	Mean 4.24 / 5. Interface and design rated highest (4.73)

Model performance (repeated 5×5 cross-validation, log scale):

Model	Mean R²	Mean RMSE	Mean MAE
Random Forest	0.5956	2.8155	2.2237
Elastic Net	0.5361	2.9305	2.3503
Ridge	0.5286	2.9366	2.3407
MLR	0.5108	2.9405	2.2686
Lasso	0.4961	3.0044	2.3995
Baseline (median)	−0.0810	4.7002	3.9539

Tier classification on the 20-country test set:

Model	Accuracy	Macro F1
Random Forest	.90	.90
Lasso	.60	.58
MLR	.55	.50
Elastic Net	.55	.54
Ridge	.50	.46
Baseline (median)	.45	.21

Random Forest classified 4 of 5 Low, 8 of 9 Medium, and 6 of 6 High countries correctly. An "always predict Medium" classifier would already score 45% on this test set, so the gain is not just the majority class.

Notable patterns:

Raw case counts are extremely right-skewed (median 25,890 vs. mean ~1.57 million), so models were trained on a log1p-transformed target.
Lasso shrank six terms to exactly zero, including GDP, urbanization, and four WHO region indicators.
Adding total population as a sensitivity check left the sanitation and Europe findings almost unchanged. Total population itself was positively associated with cases (B = 0.664, p = .034), and Random Forest improved further (cross-validated R² 0.596 → 0.649).

Read these results honestly: Random Forest's lead over the linear models in R² is smaller than its own fold-to-fold variability, and the 90% accuracy comes from a single 20-country test set (each country is worth 5 percentage points). Predicted counts are rough, with a typical error of about 17× on the original scale, which is exactly why the app reports tiers rather than exact counts. The results are best read as promising, not definitive.

3. Dashboard / Application / Demo
Streamlit application: https://malaria-prediction-tool.streamlit.app/
UI mockup (Canva): https://canva.link/rp7bbex0mnm60k5
Tableau Public: not applicable to this project

What the app does: The user enters the seven predictors (WHO region, GDP per capita, population density, urban %, sanitation access %, rainfall, temperature) and clicks Predict. The app returns the estimated case count and a color-coded tier with a plain-language description. It blocks invalid input with an error message and flags out-of-range values with a "prediction may be less reliable" warning.

Screenshots:

<!-- Replace the paths below with your actual screenshot files -->

Default state Prediction result with tier Validation error and out-of-range warning

Hosted on Streamlit Community Cloud, which sleeps after inactivity. If the app shows a wake-up screen, click to restart it.

4. Project Questions / Objectives

Main research question: How can regression-based modeling estimate national malaria case counts from cross-sectional demographic, socioeconomic, and climatic indicators, and classify that estimate into an interpretable case-count tier?

Supporting questions:

Which climatic, socioeconomic, and demographic factors significantly predict malaria case counts across nations?
How well do regression models predict malaria case burden, and which model fits best (R², RMSE, MAE)?
How accurately does the selected model classify countries into the correct Low, Medium, or High tier?
Which factors have the greatest influence on predicted case counts?

Problems addressed:

Prior work is dominated by complex models that are hard to explain to policymakers.
Most existing systems cover a single country or region.
Climate, socioeconomic, and demographic drivers are usually studied separately rather than in one framework.
5. Data
Item	Detail
Dataset	Relative Malaria Case Count Prediction Dataset (compiled by the team)
Sources	WHO Global Health Observatory (malaria cases); World Bank (GDP per capita, population density, urban %, sanitation access, total population); Open-Meteo (rainfall, temperature); REST Countries API (capital-city coordinates used only to query Open-Meteo)
Period	2015–2024, collapsed to one median value per country
Records	106 countries collected, 99 used (one row per country)
Predictors	7 (WHO region is one-hot encoded, giving 11 model terms)
Target	Median_Malaria_Cases (log1p-transformed for modeling)
Derived output	Case_Count_Tier (Low / Medium / High)

Limitations:

Small sample: 99 countries, about 20 per test set, so metrics are sensitive to which countries fall in each fold.
Seven countries were excluded for missing data. Excluded countries tend to have weaker reporting and possibly higher burden, so findings may not generalize to them.
Climate values come from capital-city coordinates, which may not represent malaria-endemic areas. This may help explain why rainfall and temperature were not significant.
Cross-sectional snapshot: it cannot capture trends over time.
About 40% of the variance in log cases is not captured by the seven predictors.
6. Data Preparation
Data quality checks: Shapiro-Wilk tests on each country's yearly series (reported for transparency only, since ten observations per series gives low power) and a VIF multicollinearity screen (all VIFs below 2).
Missing values: Countries missing any study variable (or WHO malaria data) were excluded (7 of 106) rather than imputed. Median aggregation reduces the effect of missing or anomalous individual years.
Aggregation: Ten annual values collapsed to one median per country, giving one row per country. Temperature was first averaged from daily to annual means; rainfall was summed to yearly totals.
Data types: WHO region treated as categorical; all other predictors continuous. Coordinates serve as merge and query keys only.
Feature engineering: Median aggregation, log1p transformation of the target, and quartile-based tier labels.
Encoding and transformations: One-hot encoding for WHO region; standardization of numeric predictors; tier thresholds Q1 and Q3 computed from the raw training counts only. Encoding, scaling, and thresholds are all fit on the training set to avoid leakage.
Features investigated and rejected: ITN coverage (insufficient country coverage, missing values concentrated in high-burden countries) and health expenditure per capita (VIF ≈ 13 with GDP).
7. Analysis & Methodology
Exploratory analysis: Distribution checks (strong right skew in cases), correlation analysis (Pearson and Spearman), VIF screening.
Statistical analysis: OLS regression on the log-transformed target across all 99 countries to test predictor significance.
Models compared: Median baseline, MLR, Ridge, Lasso, Elastic Net, and Random Forest (added as a non-linear benchmark).
Evaluation: 80/20 split (79 train / 20 test); alpha for Ridge, Lasso, and Elastic Net chosen by inner 5-fold CV on training data; repeated 5×5 cross-validation for model comparison; R², RMSE, and MAE on the log scale.
Tier classification: Predictions are converted back with expm1 and compared against raw-scale training quartiles (Low < Q1, Medium from Q1 to Q3 inclusive, High > Q3). Evaluated with accuracy, confusion matrices, and per-tier precision, recall, and F1.
Sensitivity analysis: Total population added as an eighth predictor as a robustness check. The seven-predictor model remains the final deployed model.
Visualization: Cross-validated metric bar chart and confusion matrices for each model.
Testing: Nine system test cases (including exact Q1/Q3 boundary checks and parity with the notebook within ±0.0001) plus usability testing with three stakeholders.
8. Key Insights
Sanitation is the signal that holds up. Countries with better sanitation access had lower predicted case counts, and this held in OLS, in all four linear models, in Random Forest importance, and after controlling for total population. This is an association, not proof of cause.
Region matters more than climate here. European countries had significantly lower case counts than the African reference region. Rainfall and temperature were not significant at the country-median level, which does not contradict studies that found climate effects at local or seasonal scales.
Simple models stay competitive. MLR had the best metrics on the single 80/20 split and trailed Random Forest by only 0.085 mean R² in cross-validation. Random Forest's real edge was in tier classification, especially catching all High-burden countries.
Tiers are more honest than exact counts. With typical errors around 17× on the original scale, a Low / Medium / High label is a defensible output where a single number would not be.
Interpretability has to reach the interface. Usability participants could operate the tool easily but asked for plain-language tier explanations, which were added.
9. Recommendations

For public health and research use:

Use the tiers for rough prioritization and triage, not for exact case forecasting.
Treat sanitation access as a priority variable to monitor alongside malaria burden, while recognizing the association is not causal evidence.

For future work:

Expand country coverage as WHO reporting improves, and test additional models (e.g., gradient boosting and stacking).
Replace capital-city climate with population-weighted climate values.
Add total population as a predictor and consider incidence per 100,000 as the outcome.
Add subnational and time-series views, which users asked for.

For the application:

Add a color-coded overview of all 99 countries and a CSV batch-prediction feature.
Refine field labels and tooltips (the lowest-rated usability item).
Move to a containerized, always-on deployment (e.g., Docker) and add a retraining pipeline for when WHO and World Bank data update.
10. Tools & Technologies
Language: Python 3.10
Data and statistics: pandas, NumPy, SciPy, statsmodels
Machine learning: scikit-learn (Pipelines, ColumnTransformer, OneHotEncoder, StandardScaler, TransformedTargetRegressor), joblib
Visualization: matplotlib, seaborn; Canva (UI mockup)
Application and deployment: Streamlit, Streamlit Community Cloud
Dependency management: pip-tools (requirements.in compiled to requirements.txt)
Development environment: Jupyter Notebooks in VS Code, conda, Git and GitHub
11. Workflow
Collect data (WHO, World Bank, Open-Meteo, REST Countries; 2015-2024)
  -> Clean and merge; median-aggregate to one row per country
  -> Exclude countries with missing variables (106 -> 99)
  -> Screen predictors (VIF) and test significance (OLS on all 99)
  -> 80/20 split; fit encoding, scaling, and Q1/Q3 on training data only
  -> Train and tune 6 models; compare with repeated 5x5 CV
  -> Select best model; evaluate tier accuracy on the held-out 20 countries
  -> Serialize pipeline and thresholds (.pkl via joblib)
  -> Deploy Streamlit app (input validation, prediction, tier display)
  -> System testing (9 cases) and usability testing (3 participants)
12. Project Structure
<!-- Adjust folder and file names to match the actual repository -->
.
├── app/
│   ├── app_ui.py              # Streamlit entry point
│   └── validation.py          # Input validation (hard errors and range warnings)
├── models/
│   ├── <one pipeline .pkl per model>
│   └── tier_thresholds.pkl    # Q1 and Q3 from training data
├── data/                      # Compiled country-level dataset
├── notebooks/                 # Data prep, EDA, modeling, evaluation
├── docs/images/               # README screenshots
├── requirements.in
├── requirements.txt           # Compiled with pip-tools
└── README.md
13. Conclusion

Summary: Using 99 countries and seven public indicators, this project shows that regression-based modeling can give an interpretable, multi-country view of malaria burden. A Random Forest, deployed as a Streamlit tool, assigned 90% of test countries to the correct Low, Medium, or High tier and identified every High-burden country in the test set.

Main takeaway: Sanitation access is the most consistent factor associated with lower malaria burden across models, and reporting results as tiers fits the precision the data can actually support.

Overall contribution: It fills a gap in multi-country, integrated-indicator, explainable malaria prediction, and it delivers a working, tested, user-evaluated prototype. The results are promising rather than definitive because of the small dataset, and the paper states that openly.

14. Author

David Zachary Ilaigilag Malonjao BSIT Data Analytics student, Mapúa Malayan Digital College

Team: Caylyne Janae Lucero, Jose Martin Marquez, Jovelyn Plaga Adviser: John Edmon Alcomendas

Skills demonstrated: end-to-end data pipeline building (collection, cleaning, merging, modeling, deployment), regression and ensemble modeling, statistical inference (OLS, VIF, normality testing), leakage-aware evaluation, Streamlit app development with input validation, software and usability testing, and team-based project delivery.
