<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=Global%20Weather%20Forecasting&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=Advanced%20ML%20%7C%20Deep%20Learning%20%7C%20Time-Series%20Analysis&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Best_Model-success?style=for-the-badge&logo=leaf&logoColor=white)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boost-blue?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-FF7043?style=for-the-badge)](https://facebook.github.io/prophet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> ### 🏢 PM Accelerator Mission
> *Accelerating the careers of product managers worldwide through real-world experience,*
> *mentorship, and community — empowering them to build impactful products.*

<br/>

</div>

---

## 🎬 Demo Video

<div align="center">

[![Demo Video](https://img.shields.io/badge/▶%20Watch%20Full%20Demo-Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1aolq8NlMXdwedUVeeONcaL6NG8BaBd7W/view?usp=drive_link)

*End-to-end walkthrough: data cleaning → EDA → anomaly detection → 9 models → leaderboard (~2 min)*

</div>

---

## 📊 Project at a Glance

<div align="center">

| 📁 Dataset | 🗓️ Date Range | 🎯 Target | 🏆 Best Model |
|:---:|:---:|:---:|:---:|
| 143,262 rows × 41 features | May 2024 → May 2026 | `temperature_celsius` | LightGBM |

| 📐 Engineered Features | 🤖 Models Built | 🔍 Anomalies Found | ✅ Best R² |
|:---:|:---:|:---:|:---:|
| 59 total (18 new) | 9 across 3 tiers | 610 consensus | **0.961** |

</div>

---

## 🗂️ Repository Structure

```
🌍 global-weather-forecasting/
│
├── 📓 GlobalWeather_Advanced_Assessment.ipynb   ← Main notebook (run this)
├── 📋 requirements.txt                          ← All Python dependencies  
├── 📖 README.md                                 ← This file
└── 📊 GlobalWeatherRepository.csv              ← Dataset (download from Kaggle)
```

---

## ⚙️ Quick Start

```bash
# 1 — Clone
git clone https://github.com/YOUR_USERNAME/global-weather-forecasting.git
cd global-weather-forecasting

# 2 — Install dependencies
pip install -r requirements.txt

# 3 — Download dataset → place GlobalWeatherRepository.csv in root
#     https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

# 4 — Launch
jupyter notebook GlobalWeather_Advanced_Assessment.ipynb
# Kernel → Restart & Run All
```

> 💡 **GPU users:** swap `tensorflow` for `tensorflow-gpu` in `requirements.txt` for 3–5× faster LSTM/Transformer training.

---

## 🔬 Full Pipeline

```
Raw CSV (143K rows)
      │
      ▼
┌─────────────────────┐
│   Data Cleaning     │  Missing values · IQR capping · DateTime parsing
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Advanced EDA       │  Distributions · Correlations · Decomposition
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Anomaly Detection   │  Isolation Forest + LOF → 610 consensus anomalies
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Engineering │  18 new features: lags · rolling stats · heat index
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Climate & Spatial  │  Country rankings · Geo-scatter map (Plotly)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Air Quality EDA   │  PM2.5 · CO · NO₂ vs weather parameters
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Importance  │  Random Forest MDI + SHAP (XGBoost)
└──────────┬──────────┘
           │
      ┌────┴────┐
      ▼         ▼
  Classical    ML & DL
  (ARIMA,    (XGBoost, LightGBM,
  Prophet)    RF, GBM, LSTM,
              Transformer)
      │         │
      └────┬────┘
           ▼
┌─────────────────────┐
│  Weighted Ensemble  │  Inverse-MAE weighting → best overall accuracy
└─────────────────────┘
```

---

## 🧹 1 · Data Cleaning & Preprocessing

| Step | Action | Result |
|------|--------|--------|
| DateTime parsing | `pd.to_datetime()` on `last_updated` | Chronological sort |
| Missing values | Per-country median → global median fallback | 0 columns dropped |
| Outlier capping | IQR 1st–99th percentile on 6 core features | Extremes suppressed |
| Shape | 143,262 × 41 → **143,262 × 59** after engineering | Clean & enriched |

---

## 📈 2 · Exploratory Data Analysis

**Key correlations discovered:**

```
temperature_celsius  ↔  feelslike_c          →  r = 0.99  (near-perfect)
humidity             ↔  precip_mm            →  r = 0.54  (moderate positive)
pressure_mb          ↔  wind_kph             →  r = −0.21 (weak negative)
uv_index             ↔  temperature_celsius  →  r = 0.47  (moderate positive)
```

**ADF Stationarity Test:**
```
ADF Statistic : −1.237
p-value       :  0.658   → Non-stationary ⚠️  (d=1 differencing applied for ARIMA)
```

---

## 🚨 3 · Anomaly Detection

Two independent unsupervised methods — an anomaly is flagged only when **both agree**:

```
Method                              Contamination    Flagged
──────────────────────────────────────────────────────────
Isolation Forest                       3.0 %          4,298
Local Outlier Factor  (k=20)           3.0 %          4,298
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Consensus  (both agree)                  —              610  ✅
```

> 📌 Anomalies cluster around extreme seasonal spikes and abnormal inter-day temperature jumps — not noise.

---

## ⚙️ 4 · Feature Engineering

18 new features engineered from the original 41:

| Category | Features Created |
|----------|-----------------|
| **Temporal** | `year`, `month`, `day_of_year`, `week`, `season` |
| **Rolling averages** | `temp_roll7`, `temp_roll30`, `humidity_roll7`, `precip_roll7` |
| **Lag features** | `temp_lag1`, `temp_lag3`, `temp_lag7` |
| **Derived metrics** | `heat_index`, `feels_delta` |

---

## 🌍 5 · Climate & Spatial Analysis

- **Hottest countries:** Mali, Djibouti, Burkina Faso, Niger, Senegal
- **Coldest countries:** Russia (Siberia), Canada, Norway, Iceland, Mongolia
- **Interactive geo-scatter map** (`geo_map.html`) — temperature + humidity overlaid on world map via Plotly

---

## 💨 6 · Air Quality — Environmental Impact

Air quality columns analysed:

```
air_quality_Carbon_Monoxide    air_quality_Ozone
air_quality_Nitrogen_dioxide   air_quality_Sulphur_dioxide
air_quality_PM2.5              air_quality_PM10
air_quality_us-epa-index       air_quality_gb-defra-index
```

**Finding:** PM2.5 and CO *rise* with temperature and *fall* with wind speed — stagnant hot air traps pollutants. Precipitation shows the strongest negative correlation with all pollutants.

---

## 🎯 7 · Feature Importance

### SHAP Analysis (XGBoost)

```
Rank  Feature                          Importance
────────────────────────────────────────────────
 1    temperature_celsius_roll7         ████████████  (highest)
 2    temp_lag1                         ███████████
 3    day_of_year                       █████████
 4    temperature_celsius_roll30        ███████
 5    temp_lag3                         ██████
 6    temp_lag7                         █████
 7    month                             ████
 8    humidity                          ███
 9    pressure_mb                       ██
10    uv_index                          █
```

> Rolling-7 temperature + yesterday's temperature alone explain **>60%** of predictive power.

---

## 🤖 8 · Models Built

### Tier 1 — Classical Forecasting

| Model | Config | MAE | RMSE | R² |
|-------|--------|-----|------|----|
| ARIMA | order=(2,1,2) | 1.084 | 1.240 | −2.931* |
| Prophet | yearly + weekly seasonality | 0.640 | 0.746 | −0.423* |

*\*Evaluated on 30-day univariate series — different test set from ML models.*

---

### Tier 2 — Machine Learning (113K train / 28K test, chronological split)

| Model | Trees | Depth | MAE | RMSE | R² |
|-------|-------|-------|-----|------|----|
| Random Forest | 200 | 12 | 1.490 | 2.063 | 0.957 |
| Gradient Boosting | 300 | 5 | 1.489 | 2.032 | 0.958 |
| XGBoost | 300 | 6 | 1.453 | 1.982 | 0.960 |
| **LightGBM** ⭐ | **500** | **8** | **1.431** | **1.955** | **0.961** |

---

### Tier 3 — Deep Learning (look-back=30 days → horizon=7 days)

| Model | Architecture | MAE | RMSE | R² |
|-------|-------------|-----|------|----|
| LSTM | 128→64 units, Dropout 0.2 | 1.830 | 2.484 | 0.937 |
| Temporal Transformer | 4 heads, d_model=64, FFN=128 | 1.923 | 2.610 | 0.931 |

**LSTM architecture:**
```
Input(30, 14) → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(7)
```

**Transformer architecture:**
```
Input(30, 14) → Dense(64) → PosEncoding → MultiHeadAttention(4h) → LayerNorm
             → FFN(128) → LayerNorm → GlobalAvgPool → Dense(64) → Dense(7)
```

---

## 🏆 9 · Final Leaderboard

```
╔══════╦═══════════════════════════╦═════════╦═════════╦═════════╗
║ Rank ║ Model                     ║   MAE   ║  RMSE   ║   R²    ║
╠══════╬═══════════════════════════╬═════════╬═════════╬═════════╣
║  🥇  ║ LightGBM                  ║  1.431  ║  1.955  ║  0.961  ║
║  🥈  ║ Ensemble (Weighted)       ║  1.446  ║  1.982  ║  0.960  ║
║  🥉  ║ XGBoost                   ║  1.453  ║  1.982  ║  0.960  ║
║   4  ║ Gradient Boosting         ║  1.489  ║  2.032  ║  0.958  ║
║   5  ║ Random Forest             ║  1.490  ║  2.063  ║  0.957  ║
║   6  ║ LSTM                      ║  1.830  ║  2.484  ║  0.937  ║
║   7  ║ Temporal Transformer      ║  1.923  ║  2.610  ║  0.931  ║
║   8  ║ Prophet *                 ║  0.640  ║  0.746  ║ −0.423  ║
║   9  ║ ARIMA *                   ║  1.084  ║  1.240  ║ −2.931  ║
╚══════╩═══════════════════════════╩═════════╩═════════╩═════════╝
* Univariate 30-day test only
```

**Ensemble weights (inverse-MAE):**

```
LightGBM          25.6%  ████████████▌
XGBoost           25.2%  ████████████▎
Gradient Boosting 24.6%  ████████████
Random Forest     24.6%  ████████████
```

---

## 💡 Key Insights

| # | Insight |
|---|---------|
| 1 | **Rolling-7 temperature is the #1 predictor** — recent history dominates over all static weather features |
| 2 | **610 consensus anomalies** align with documented regional heatwaves — the detection is meaningful, not noise |
| 3 | **LightGBM beats deep learning** on this dataset — sufficient tabular data means gradient boosting still wins |
| 4 | **PM2.5 & CO track with temperature** — environmental policy implications for hot-climate urban planning |
| 5 | **Ensemble adds marginal gain** — individual model diversity was low; more heterogeneous architectures would help |
| 6 | **LSTM outperforms Transformer** on short horizons — attention advantage appears beyond 14-day forecasting |

---

## 🔭 Future Work

- [ ] Hyperparameter tuning with **Optuna** for LightGBM & Transformer
- [ ] **Multi-target forecasting** — jointly predict temperature + humidity + precipitation
- [ ] **Probabilistic forecasting** — quantile regression or MC-Dropout for uncertainty bands
- [ ] **Real-time pipeline** — OpenWeatherMap API stream + incremental retraining
- [ ] **Deployment** — FastAPI endpoint + Streamlit dashboard

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Tools |
|-------|-------|
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) |
| **Viz** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-76b900?style=flat) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) |
| **ML** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-189fdd?style=flat) ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat) ![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=flat) |
| **Time-Series** | ![Statsmodels](https://img.shields.io/badge/Statsmodels-3d6cb9?style=flat) ![Prophet](https://img.shields.io/badge/Prophet-FF7043?style=flat) |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) |
| **Notebook** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) |

</div>

---

<div align="center">

**Assessment submitted for PM Accelerator — Data Scientist / Analyst Role**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Open Source](https://img.shields.io/badge/Open-Source-brightgreen?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%"/>

</div>
