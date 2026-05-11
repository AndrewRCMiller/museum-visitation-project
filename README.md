# Museum Visitation Machine Learning Project

SEIS 763 Machine Learning project focused on forecasting monthly museum visitation using museum attendance data, weather data, Google Trends demand signals, calendar/seasonal features, and lag-based visitor features.

---

## Overview

This project builds a machine learning workflow to predict **monthly museum visitor counts**. The project combines multiple data sources to simulate a real-world forecasting problem where a museum could use visitor predictions to support staffing, operations, marketing timing, and planning.

The final modeling dataset includes:

- Museum attendance data
- Weather data
- Google Trends search-interest data
- Calendar and seasonal features
- Lag features based on prior visitor counts

---

## Project Goals

- Forecast monthly museum visitation using machine learning
- Understand the main drivers of attendance
- Evaluate whether weather influences visitation
- Use Google Trends as a public-interest and demand proxy
- Build a reproducible workflow from raw data to final modeling results
- Compare multiple machine learning models and select the best-performing approach

---

## Project Structure

### Root Files

- `README.md` — project overview and workflow notes
- `requirements.txt` — Python package requirements
- `.gitignore` — ignored local files and folders

---

### data/

#### raw/

- `museum/` — raw museum attendance files
- `weather/` — raw weather files
- `trends/` — raw Google Trends files
- `calendar/` — raw calendar feature files

#### processed/

- `museum_monthly.csv` — cleaned monthly museum visitation data
- `weather_monthly.csv` — cleaned monthly weather data
- `calendar_monthly.csv` — calendar and seasonal features
- `final_dataset.csv` — merged project dataset
- `model_dataset.csv` — final modeling dataset with lag features
- `google-trends-marketing-v2/` — processed Google Trends marketing datasets
- `python_v1/` — versioned Google Trends workflow outputs

---

### notebooks/

- `01_museum_data.ipynb` — museum attendance preparation
- `02_weather_data.ipynb` — weather data preparation
- `03_google_trends.ipynb` — Google Trends data work
- `04_calendar_features.ipynb` — calendar and seasonal features
- `05_feature_engineering.ipynb` — feature creation and preparation
- `06_modeling.ipynb` — machine learning modeling and evaluation
- `07_presentation_visuals.ipynb` — visuals for final presentation
- `08_build_final_dataset.ipynb` — final dataset merge workflow
- `09_create_model_dataset.ipynb` — creation of model-ready dataset with lag features

---

### src/

- `google_trends_marketing/` — Google Trends workflow scripts
- `fetch_google_trends.py` — fetch Google Trends data
- `rank_trends_keywords.py` — rank Google Trends keyword opportunities
- `summarize_trends_insights.py` — summarize Google Trends insights and plots
- `run_trends_versioned.py` — run versioned Google Trends pipeline
- `import_google_trends_exports.py` — import Google Trends exports

---

### outputs/

#### plots/

- Modeling plots
- Google Trends marketing plots
- Presentation-ready visuals

#### tables/

- Model comparison table
- Feature importance tables
- Modeling outputs

#### models/

- Saved model artifacts, if used

---

### docs/

- `proposal.docx` — project proposal
- `final_notes.md` — final project notes
- `google_trends_marketing/` — Google Trends documentation, notes, sources, and runbooks

---

## Data Sources

### Museum Visitor Data

- Monthly museum visitor counts
- Target variable: `visitors`
- Used to train and evaluate forecasting models

### Weather Data

Weather features include:

- Average temperature
- Minimum temperature
- Maximum temperature
- Total precipitation
- Average wind speed

Weather values were converted to U.S. units where needed:

- Temperature: Fahrenheit
- Precipitation: inches
- Wind speed: miles per hour

### Google Trends Data

Google Trends was used as a proxy for public interest and tourism demand. Search terms included attraction, museum, local history, and tourism-related phrases.

Examples include:

- `los_angeles_attractions`
- `los_angeles_museums`
- `olvera_street`
- `el_pueblo_los_angeles`
- `getty_museum`
- `things_to_do_in_la`

The Google Trends workflow includes versioned outputs and supporting documentation in:

- `src/google_trends_marketing/`
- `docs/google_trends_marketing/`
- `outputs/plots/google_trends_marketing/`
- `data/processed/python_v1/`

### Calendar and Seasonal Features

Calendar variables include:

- Month-based seasonality
- Holiday indicators
- School break indicators
- Seasonal flags
- Cyclical month features such as `month_sin` and `month_cos`

---

## Data Workflow

1. Build individual monthly datasets for museum, weather, Google Trends, and calendar features.
2. Ensure all datasets use a shared `month` field.
3. Merge datasets into `final_dataset.csv`.
4. Create lag and rolling features.
5. Save final modeling file as `model_dataset.csv`.
6. Use `model_dataset.csv` for machine learning.

Example merge logic:

```python
df = museum.merge(weather, on="month", how="left")
df = df.merge(trends, on="month", how="left")
df = df.merge(calendar, on="month", how="left")
```

---

## Feature Engineering

The final model dataset includes lag-based and seasonal features:

- `visitors_lag1` — previous month visitor count
- `visitors_lag12` — same month visitor count from the previous year
- `rolling_mean_3` — rolling three-month average of visitors
- `month_sin` and `month_cos` — cyclical month features
- Holiday and seasonal indicators
- Weather variables
- Google Trends search-interest variables

The lag features are important because they allow the model to learn short-term momentum and yearly seasonality in museum visitation.

---

## Modeling Approach

The modeling notebook compares several regression models:

- Linear Regression
- Cleaned Linear Regression
- LASSO
- LASSO-LARS
- Random Forest Regression
- Gradient Boosting Regression
- SVR with RBF kernel
- SVR with linear kernel

### Final Model Selection

The best-performing models were **LASSO-LARS** and **LASSO**. These models performed well because they reduce overfitting and handle correlated predictors by shrinking weaker variables toward zero.

### Model Evaluation Metrics

Models were evaluated using:

- RMSE — Root Mean Squared Error
- MAE — Mean Absolute Error
- R² Score

### Final Model Comparison

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

## Key Findings

- Lag features were among the strongest predictors of museum visitation.
- `visitors_lag1` captured short-term visitor momentum.
- `visitors_lag12` captured yearly seasonality.
- Google Trends variables helped represent public interest and tourism demand.
- Weather variables contributed to the model, but they were not the strongest predictors.
- LASSO and LASSO-LARS outperformed more complex tree-based models.
- The RBF SVR model underperformed, while the tuned linear SVR performed reasonably well.
- Regularized linear models provided the best balance of accuracy and interpretability.

---

## COVID-19 Handling

COVID-19 closure months were removed before modeling because zero visitor counts during forced closure do not represent normal visitation behavior. Removing these months helped prevent the model from learning an artificial closure pattern.

The removed period was:

- April 2020 through May 2021

---

## How to Run

### Clone the Repository

```bash
git clone https://github.com/Andy-FireClimWx/museum-ml-project.git
cd museum-ml-project
```

### Create Environment

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

### Install Packages

```bash
pip install -r requirements.txt
```

### Run the Notebook Workflow

Run the notebooks in order:

1. `01_museum_data.ipynb`
2. `02_weather_data.ipynb`
3. `03_google_trends.ipynb`
4. `04_calendar_features.ipynb`
5. `05_feature_engineering.ipynb`
6. `08_build_final_dataset.ipynb`
7. `09_create_model_dataset.ipynb`
8. `06_modeling.ipynb`
9. `07_presentation_visuals.ipynb`

---

## Google Trends Workflow

Build Google Trends data:

```bash
python src/google_trends_marketing/fetch_google_trends.py
```

Use cached raw data if rate-limited:

```bash
python src/google_trends_marketing/fetch_google_trends.py --use-existing-raw
```

Resume from existing raw data:

```bash
python src/google_trends_marketing/fetch_google_trends.py --resume-existing-raw
```

Build keyword ranking:

```bash
python src/google_trends_marketing/rank_trends_keywords.py
```

Print summaries and save plots:

```bash
python src/google_trends_marketing/summarize_trends_insights.py
```

Run a versioned workflow:

```bash
python src/google_trends_marketing/run_trends_versioned.py --run-label python_v1 --use-existing-raw
```

---

## Team Roles

- Project Lead
- Museum Data
- Weather Data
- Google Trends Data
- Calendar Features
- Feature Engineering
- Modeling
- Presentation

---

## Final Conclusion

This project shows that monthly museum visitation can be forecasted reasonably well using historical attendance patterns, seasonal features, Google Trends demand signals, and weather data. The strongest models were LASSO and LASSO-LARS, which performed best because they reduced overfitting and handled correlated predictors effectively. The most important predictors were lag features and public-interest variables, showing that past attendance and search behavior are stronger signals than weather alone. Overall, the project demonstrates a complete machine learning workflow from data collection and feature engineering to model comparison and interpretation.

---

## Notes

- All datasets should use `month` format: `YYYY-MM`.
- The model should use `model_dataset.csv`, not only `final_dataset.csv`.
- Missing values are handled after merging.
- COVID closure months are removed before modeling.
- Output plots and tables are saved under the root `outputs/` folder.
