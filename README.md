# 🏦 SBA 7(a) Loan Default Predictor

An interactive ML-powered dashboard for analyzing SBA 7(a) loan default risk (FY2010–FY2019), built with **Streamlit**, **scikit-learn**, and **XGBoost**.

## Features

- 📊 **Portfolio Overview** — KPI cards, summary table, donut chart, sector default rates
- 🗺️ **Geographic Risk** — State-level choropleth maps (count vs. dollar default rate)
- 🤖 **ML Models** — Random Forest & XGBoost training, ROC curves, confusion matrices, optimal threshold analysis
- 📈 **Feature Importance** — XGBoost gain-based ranking of all loan features
- 💾 **Model Download** — Export trained pipeline as `.pkl`

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data

Upload the FOIA SBA 7(a) CSV dataset via the sidebar (e.g. `foia-7a-fy2010-fy2019-asof-*.csv`).  
Dataset available from the [SBA FOIA Reading Room](https://www.sba.gov/about-sba/sba-performance/open-government/foia/reading-room-meta/resources-small-businesses).

## Tech Stack

| Layer | Library |
|---|---|
| App | Streamlit |
| ML | scikit-learn, XGBoost |
| Viz | Plotly, Matplotlib |
| Data | Pandas, NumPy |
