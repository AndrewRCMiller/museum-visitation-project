# Museum Visitation Machine Learning Project

SEIS 763 Machine Learning project focused on forecasting monthly museum visitation using museum attendance data, weather data, Google Trends demand signals, calendar/seasonal features, and lag-based visitor features.

---

# Overview

This project builds a complete machine learning workflow to predict **monthly museum visitor counts**. The project combines multiple data sources to simulate a real-world forecasting problem where museums and historic sites could use visitor predictions to support staffing, operations, marketing timing, budgeting, and long-term planning.

The workflow includes:

- Data collection
- Data cleaning
- Feature engineering
- Time-series forecasting
- Machine learning model comparison
- Visualization and interpretation

The final modeling dataset includes:

- Museum attendance data
- Weather data
- Google Trends search-interest data
- Calendar and seasonal features
- Lag features based on prior visitor counts

---

# Project Goals

- Forecast monthly museum visitation using machine learning
- Understand the main drivers of attendance
- Evaluate whether weather influences visitation
- Use Google Trends as a public-interest and demand proxy
- Build a reproducible workflow from raw data to final modeling results
- Compare multiple machine learning models and select the best-performing approach

---

# Project Structure

```text
museum-visitation-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── museum/
│   │   ├── weather/
│   │   ├── trends/
│   │   └── calendar/
│   │
│   └── processed/
│       ├── museum_monthly.csv
│       ├── weather_monthly.csv
│       ├── calendar_monthly.csv
│       ├── final_dataset.csv
│       ├── model_dataset.csv
│       ├── google-trends-marketing-v2/
│       └── python_v1/
│
├── notebooks/
│   ├── 01_museum_data.ipynb
│   ├── 02_weather_data.ipynb
│   ├── 03_google_trends.ipynb
│   ├── 04_calendar_features.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_modeling.ipynb
│   ├── 07_presentation_visuals.ipynb
│   ├── 08_build_final_dataset.ipynb
│   └── 09_create_model_dataset.ipynb
│
├── outputs/
│   ├── plots/
│   ├── tables/
│   └── models/
│
├── src/
│   ├── google_trends_marketing/
│   ├── fetch_google_trends.py
│   ├── rank_trends_keywords.py
│   ├── summarize_trends_insights.py
│   ├── run_trends_versioned.py
│   └── import_google_trends_exports.py
│
└── docs/
    ├── Proposal.docx
    ├── Final_Report.docx
    ├── Presentation.pptx
    ├── final_notes.md
    └── google_trends_marketing/
```

---

# Data Sources

## Museum Visitor Dataset

Monthly museum visitor counts for Los Angeles museums from 2014–2021.

Project focus:
- Avila Adobe historic site

Dataset source:
https://www.kaggle.com/code/yasinnaal/los-angeles-museums-visitors

---

## NOAA Climate Data

Monthly weather data including:

- Average temperature
- Minimum temperature
- Maximum temperature
- Total precipitation
- Average wind speed

Dataset source:
https://www.ncdc.noaa.gov/cdo-web/datasets

---

## Holiday and Calendar Features

Calendar-based features include:

- Holiday indicators
- School break indicators
- Tourism seasons
- Seasonal categories
- Cyclical month features

Dataset source:
https://pypi.org/project/holidays/

---

## Google Trends Data

Google Trends was used as a proxy for public interest and tourism demand.

Example search terms:

- los_angeles_museums
- things_to_do_in_la
- olvera_street
- getty_museum
- los_angeles_attractions
- el_pueblo_los_angeles

Dataset source:
https://trends.google.com/trends/

The Google Trends workflow includes:

- Versioned outputs
- Marketing plots
- Keyword ranking
- Supporting documentation

Related folders:

- src/google_trends_marketing/
- docs/google_trends_marketing/
- outputs/plots/google_trends_marketing/

---

# Data Workflow

1. Build monthly museum dataset
2. Build monthly weather dataset
3. Build Google Trends dataset
4. Build calendar/seasonality dataset
5. Merge all datasets into final_dataset.csv
6. Create lag and rolling features
7. Save final modeling file as model_dataset.csv
8. Train and compare machine learning models

Example merge logic:

```python
df = museum.merge(weather, on="month", how="left")
df = df.merge(trends, on="month", how="left")
df = df.merge(calendar, on="month", how="left")
```

---

# Feature Engineering

The final modeling dataset includes:

## Lag Features

- visitors_lag1
- visitors_lag12
- rolling_mean_3

These features help the model learn:

- Short-term visitor momentum
- Yearly seasonality patterns
- Attendance trends

---

## Weather Features

- avg_temp_F
- min_temp_F
- max_temp_F
- total_precip_in
- avg_wind_mph

Weather values were converted into U.S. units where necessary.

---

## Seasonal Features

- is_summer
- is_winter
- holiday indicators
- spring_break
- summer_tourism_season
- month_sin
- month_cos

---

## Google Trends Features

Google Trends variables represent:

- Tourism interest
- Public awareness
- Search demand
- Attraction popularity

---

# Modeling Approach

The project compares multiple regression models:

- Linear Regression
- Cleaned Linear Regression
- LASSO Regression
- LASSO-LARS
- Random Forest Regression
- Gradient Boosting Regression
- SVR (RBF Kernel)
- SVR (Linear Kernel)

---

# Evaluation Metrics

Models were evaluated using:

- RMSE — Root Mean Squared Error
- MAE — Mean Absolute Error
- R² Score

---

# Final Model Comparison

| Rank | Model | RMSE | MAE | R² |
|---:|---|---:|---:|---:|
| 1 | LASSO-LARS | 2982.044 | 2259.141 | 0.783 |
| 2 | LASSO | 2986.711 | 2258.057 | 0.782 |
| 3 | SVR (Linear) | 3368.630 | 2368.915 | 0.723 |
| 4 | Gradient Boosting | 5001.671 | 3340.323 | 0.389 |
| 5 | Random Forest | 5533.448 | 3604.140 | 0.252 |
| 6 | SVR (RBF) | 7229.172 | 5234.714 | -0.277 |
| 7 | Linear Regression | 8769.045 | 6205.897 | -0.879 |
| 8 | Cleaned Linear Regression | 8769.045 | 6205.897 | -0.879 |

---

# Best Model

Best overall model:
- **LASSO-LARS**

Why it performed best:

- Reduced overfitting
- Handled correlated predictors well
- Balanced accuracy and interpretability
- Selected the strongest predictors automatically

---

# Key Findings

- Lag features were the strongest predictors.
- visitors_lag1 captured short-term visitation momentum.
- visitors_lag12 captured yearly seasonality.
- Google Trends variables improved demand estimation.
- Weather contributed to visitation patterns but was weaker than lag features.
- LASSO-based models outperformed more complex tree-based models.
- Linear SVR performed reasonably well after tuning.
- RBF SVR underperformed and appeared to underfit the data.

---

# COVID-19 Handling

COVID-19 closure months were removed before final modeling because the forced closure period does not represent normal visitation behavior.

Removed period:

- April 2020 through May 2021

Removing these months improved model stability and predictive accuracy.

---

# General Dataset Statistics

- Regression forecasting problem
- Final modeling dataset contains engineered lag and seasonal features
- Missing values handled during preprocessing
- Features standardized where appropriate before modeling
- Time-based train/test split used for forecasting realism

---

# Visualization Outputs

The project includes:

- Actual vs Predicted plots
- Residual plots
- LASSO coefficient path plots
- Cross-validation error plots
- Random Forest feature importance plots
- Gradient Boosting feature importance plots
- Seasonal visitation trend plots

All outputs are saved under:

```text
outputs/plots/
outputs/tables/
```

---

# How to Run

## Clone Repository

```bash
git clone https://github.com/Andy-FireClimWx/museum-ml-project.git
cd museum-ml-project
```

---

## Create Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Notebook Workflow

Run notebooks in this order:

1. 01_museum_data.ipynb
2. 02_weather_data.ipynb
3. 03_google_trends.ipynb
4. 04_calendar_features.ipynb
5. 05_feature_engineering.ipynb
6. 08_build_final_dataset.ipynb
7. 09_create_model_dataset.ipynb
8. 06_modeling.ipynb
9. 07_presentation_visuals.ipynb

---

# Google Trends Workflow

Fetch Google Trends data:

```bash
python src/google_trends_marketing/fetch_google_trends.py
```

Use cached raw data:

```bash
python src/google_trends_marketing/fetch_google_trends.py --use-existing-raw
```

Resume existing workflow:

```bash
python src/google_trends_marketing/fetch_google_trends.py --resume-existing-raw
```

Build keyword rankings:

```bash
python src/google_trends_marketing/rank_trends_keywords.py
```

Generate summaries and plots:

```bash
python src/google_trends_marketing/summarize_trends_insights.py
```

Run versioned workflow:

```bash
python src/google_trends_marketing/run_trends_versioned.py --run-label python_v1 --use-existing-raw
```

---

# Team Roles

- Museum data preparation
- Weather data preparation
- Google Trends workflow
- Calendar feature engineering
- Feature engineering
- Machine learning modeling
- Visualization and presentation
- Documentation and reporting

---

# Final Conclusion

This project demonstrates that monthly museum visitation can be forecasted reasonably well using machine learning models combined with historical visitation patterns, seasonal indicators, Google Trends demand signals, and weather data.

The strongest models were LASSO and LASSO-LARS because they reduced overfitting and handled correlated predictors effectively. The most important predictors were lag variables and Google Trends public-interest features, showing that past attendance behavior and tourism demand are stronger signals than weather alone.

Overall, the project demonstrates a complete end-to-end machine learning workflow from data collection and feature engineering to model evaluation, interpretation, and forecasting.

---

# Notes

- All datasets use month format YYYY-MM
- model_dataset.csv should be used for modeling
- COVID closure months were removed before modeling
- Outputs are saved to the root outputs/ folder
- Time-based train/test splitting was used for final forecasting models
