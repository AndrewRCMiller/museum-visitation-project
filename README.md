# museum-visitation-ml
SEIS 763 ML project – Museum visitation forecasting using weather, Google Trends, and seasonality

# Museum Visitation Machine Learning Project

## Overview

This project is part of **SEIS 763: Machine Learning**. The goal is to build a machine learning model that predicts **monthly museum visitor counts** using a combination of:

- Museum attendance data  
- Weather data  
- Google Trends (as a marketing/demand proxy)  
- Calendar and seasonal features  

The project simulates a real-world data science workflow, integrating multiple data sources to generate actionable insights about visitor behavior.

---

## Objectives

- Predict monthly museum visitation using machine learning  
- Understand key drivers of attendance (weather, seasonality, demand signals)  
- Estimate the impact of public interest (Google Trends) on visitation  
- Build a reproducible and modular ML pipeline  

---

## Project Structure

```text
museum-ml-project/
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
│   └── processed/
│       ├── museum_monthly.csv
│       ├── weather_monthly.csv
│       ├── trends_monthly.csv
│       ├── calendar_monthly.csv
│       ├── final_dataset.csv
│       └── model_dataset.csv
│
├── notebooks/
│   ├── 01_museum_data.ipynb
│   ├── 02_weather_data.ipynb
│   ├── 03_google_trends.ipynb
│   ├── 04_calendar_features.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_modeling.ipynb
│   └── 07_presentation_visuals.ipynb
│
├── src/
│   ├── build_dataset.py
│   ├── features.py
│   ├── model.py
│   └── utils.py
│
├── outputs/
│   ├── plots/
│   ├── tables/
│   └── models/
│
└── docs/
    ├── proposal.docx
    ├── presentation_outline.md
    └── final_notes.md

    ---

## Data Sources

### Museum Visitor Data
- Kaggle dataset of monthly museum visitors  
- Target variable: `visitors`

### Weather Data
- Meteostat or NOAA  
- Features:
  - Average temperature  
  - Total precipitation  
  - Wind speed  

### Google Trends Data (Marketing Proxy)
- Search interest data representing public demand  
- Keywords:
  - "Los Angeles museums"
  - "things to do in Los Angeles"
  - "Los Angeles attractions"
  - "family activities Los Angeles"

### Calendar Features
- Month
- Seasonality indicators
- Holiday flags

---

## Data Workflow

1. Each team member builds a dataset  
2. Each dataset must:
   - Be monthly  
   - Include `month` column (`YYYY-MM`)  
3. Merge datasets:

```python
df = museum.merge(weather, on="month", how="left")
df = df.merge(trends, on="month", how="left")
df = df.merge(calendar, on="month", how="left")

## Feature Engineering

- `visitors_lag1` (previous month)  
- `visitors_lag12` (previous year)  
- `rolling_mean_3`  
- Seasonal features  

---

## Machine Learning Model

- Linear Regression (baseline)

### Target
- Monthly museum visitors  

### Inputs
- Weather  
- Trends  
- Calendar  
- Lag features  

---

## Evaluation Metrics

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² Score  

---

## How to Run

### Clone repo
```bash
git clone https://github.com/Andy-FireClimWx/museum-ml-project.git
cd museum-ml-project

## Setup & Execution

### Create environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate   # Mac/Linux

### Install Packages
```bash
pip install -r requirements.txt

---

## Team Roles

- Project Lead  
- Weather Data  
- Google Trends  
- Museum Data  
- Feature Engineering  
- Modeling  
- Presentation  

---

## Key Insights (Expected)

- Strong seasonal patterns  
- Weather impacts attendance  
- Google Trends reflects demand  
- Lag features improve predictions  

---

## Notes

- All datasets must use `month` format (`YYYY-MM`)  
- Missing values handled after merging  
- COVID period handled during modeling  