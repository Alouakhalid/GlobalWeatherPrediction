# 🌍 Global Weather Repository — Advanced Forecasting

> **PM Accelerator Mission:** To accelerate the careers of product managers worldwide by providing real-world experience, mentorship, and community — empowering them to build impactful products and lead high-performing teams.

---

## 📋 Project Overview

This project completes the **Advanced Track** of the PM Accelerator Data Science Technical Assessment. Using the [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) — **143,262 rows × 41 weather features** covering cities worldwide from May 2024 → May 2026 — it delivers a full production-grade forecasting pipeline.

```
143,262 records  ·  41 raw features  ·  59 engineered features
Date range: 2024-05-16 → 2026-05-24   ·   Target: temperature_celsius
```

---

## 🗂 Repository Structure

```
.
├── GlobalWeather_Advanced_Assessment.ipynb   ← Main notebook (run this)
├── requirements.txt                          ← All Python dependencies
├── README.md                                 ← This file
└── GlobalWeatherRepository.csv              ← Dataset (download from Kaggle)
```

---

## ⚙️ Setup & Installation

### 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/global-weather-forecasting.git
cd global-weather-forecasting
```

### 2 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Download dataset
Get `GlobalWeatherRepository.csv` from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) and place it in the project root.

### 5 — Run the notebook
```bash
jupyter notebook GlobalWeather_Advanced_Assessment.ipynb
# Then: Kernel → Restart & Run All
```

---

## 🔬 Methodology & Results

---

### 1 · Data Cleaning & Preprocessing

- **Dataset shape:** 143,262 rows × 41 columns — no columns dropped (0 exceeded 50 % missing threshold)
- Parsed `last_updated` as datetime, sorted chronologically
- Numeric nulls filled with **per-country median** (global median as fallback)
- Core features IQR-capped at 1st–99th percentile to suppress extremes

---

### 2 · Exploratory Data Analysis

#### 2.1 Core Feature Distributions

#### 2.2 Correlation Matrix

**Key correlations:**
- `temperature_celsius` ↔ `feelslike_c` → near-perfect (0.99)
- `pressure_mb` ↔ `wind_kph` → weak negative
- `humidity` ↔ `precip_mm` → moderate positive

#### 2.3 Daily Average Temperature Trend (2024–2026)

#### 2.4 Seasonal Decomposition (Weekly Period)

**ADF Test Result:** Statistic = −1.237, p-value = 0.658 → series is **non-stationary** → first-order differencing applied for ARIMA.

---

### 3 · Anomaly Detection

Two independent detectors applied; flagged only when **both** agree (consensus):

| Detector | Anomalies Flagged |
|---|---|
| Isolation Forest (contamination=3%) | 4,298 |
| Local Outlier Factor (k=20, contamination=3%) | 4,298 |
| **Consensus (both agree)** | **610** |

Red × markers show 610 consensus anomalies — concentrated in extreme seasonal spikes and unusual inter-day jumps.

---

### 4 · Feature Engineering

18 new features created from the raw 41:

| Category | Features |
|---|---|
| Temporal | `year`, `month`, `day_of_year`, `week`, `season` |
| Rolling averages | `temp_roll7`, `temp_roll30`, `humidity_roll7`, `precip_roll7` |
| Lag features | `temp_lag1`, `temp_lag3`, `temp_lag7` |
| Derived | `heat_index`, `feels_delta` |

Final shape after engineering: **143,262 × 59 features**

---

### 5 · Climate & Spatial Analysis

#### 5.1 Hottest & Coldest Countries

#### 5.2 Top 15 Countries by Temperature

---

### 6 · Environmental Impact — Air Quality

Air-quality columns found in dataset:
`air_quality_Carbon_Monoxide`, `air_quality_Ozone`, `air_quality_Nitrogen_dioxide`,
`air_quality_Sulphur_dioxide`, `air_quality_PM2.5`, `air_quality_PM10`,
`air_quality_us-epa-index`, `air_quality_gb-defra-index`

**Key finding:** PM2.5 and CO show positive correlation with temperature and negative correlation with wind speed — confirming that stagnant hot air traps pollutants.

---

### 7 · Feature Importance

#### 7.1 Random Forest — Mean Decrease in Impurity

#### 7.2 SHAP — Mean Absolute Impact (XGBoost)

#### 7.3 SHAP Beeswarm — Feature Impact Direction

**Top 3 most important features across all methods:**
1. `temperature_celsius_roll7` (7-day rolling average)
2. `temp_lag1` (previous day temperature)
3. `day_of_year` (cyclical seasonal signal)

---

### 8 · Classical Forecasting

#### 8.1 ACF / PACF Analysis

Slow ACF decay confirms non-stationarity. First-order differencing applied (d=1).

#### 8.2 ARIMA(2,1,2) — 30-Day Forecast

```
ARIMA  →  MAE: 1.084  |  RMSE: 1.240  |  R²: -2.931
```

#### 8.3 Prophet — Yearly + Weekly Seasonality

```
Prophet  →  MAE: 0.640  |  RMSE: 0.746  |  R²: -0.423
```

---

### 9 · Machine Learning Forecasting

Train/test split: **113,328 / 28,333 samples** (chronological, no shuffling)

```
XGBoost          →  MAE: 1.453  |  RMSE: 1.982  |  R²: 0.960
LightGBM         →  MAE: 1.431  |  RMSE: 1.955  |  R²: 0.961  ⭐ Best individual
Random Forest    →  MAE: 1.490  |  RMSE: 2.063  |  R²: 0.957
Gradient Boosting→  MAE: 1.489  |  RMSE: 2.032  |  R²: 0.958
```

---

### 10 · Deep Learning Forecasting

Sequence architecture: **look-back = 30 days → predict 7-day horizon**
Training data: **(113,300 sequences × 30 timesteps × 14 features)**

#### 10.1 LSTM Architecture
```
LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(7)
```

#### 10.2 Temporal Transformer Architecture
```
Input → Dense(64) → Positional Encoding → MultiHeadAttention(4 heads)
→ LayerNorm → FFN(128) → LayerNorm → GlobalAvgPool → Dense(64) → Dense(7)
```

#### 10.3 Training Curves

#### 10.4 Deep Learning Predictions

```
LSTM         →  MAE: 1.830  |  RMSE: 2.484  |  R²: 0.937
Transformer  →  MAE: 1.923  |  RMSE: 2.610  |  R²: 0.931
```

---

### 11 · Ensemble Model & Final Leaderboard

Weighted ensemble using **inverse-MAE weighting** of all 4 ML models:

| Model | Weight |
|---|---|
| LightGBM | 0.2560 |
| XGBoost | 0.2522 |
| Gradient Boosting | 0.2460 |
| Random Forest | 0.2458 |

```
Ensemble (Weighted)  →  MAE: 1.446  |  RMSE: 1.982  |  R²: 0.960
```

#### 11.1 All-Model Comparison

#### 📊 Final Leaderboard

| Rank | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| 🥇 | **LightGBM** | **1.431** | **1.955** | **0.961** |
| 🥈 | Ensemble (Weighted) | 1.446 | 1.982 | 0.960 |
| 🥉 | XGBoost | 1.453 | 1.982 | 0.960 |
| 4 | Gradient Boosting | 1.489 | 2.032 | 0.958 |
| 5 | Random Forest | 1.490 | 2.063 | 0.957 |
| 6 | LSTM | 1.830 | 2.484 | 0.937 |
| 7 | Transformer | 1.923 | 2.610 | 0.931 |
| 8 | Prophet | 0.640* | 0.746* | −0.423* |
| 9 | ARIMA | 1.084* | 1.240* | −2.931* |

*Classical models evaluated on 30-day univariate series only (different test set).

#### 11.2 Residual Analysis — Best Model (LightGBM)

Residuals are approximately normally distributed and centered near zero — confirming a well-calibrated model with no systematic bias.

---

## 📈 Key Insights

| Theme | Finding |
|---|---|
| **Dataset** | 143,262 records, zero columns dropped, clean from the start |
| **Anomalies** | 610 consensus anomalies; concentrated in extreme seasonal spikes |
| **Top Features** | Rolling-7 temp, lag-1 temp, and `day_of_year` drive >60% of predictive power |
| **Air Quality** | PM2.5 & CO rise with temperature, fall with wind — hot stagnant air traps pollutants |
| **Best Model** | LightGBM: MAE=1.431°C, RMSE=1.955°C, **R²=0.961** |
| **Deep Learning** | Competitive at 93.7% R² (LSTM); Transformer slightly weaker but more interpretable |
| **Stationarity** | Global avg temp series is non-stationary (ADF p=0.658); differencing required for ARIMA |

---

## 🔭 Future Work

1. **Hyperparameter tuning** — Optuna / Bayesian search for LightGBM & Transformer
2. **Multi-target forecasting** — jointly predict temperature, humidity, and precipitation
3. **Probabilistic forecasting** — quantile regression or MC-Dropout for uncertainty bands
4. **Real-time pipeline** — stream live data via OpenWeatherMap API and retrain incrementally
5. **Deployment** — wrap best model as a FastAPI endpoint with a Streamlit dashboard

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| Data | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML | Scikit-learn, XGBoost, LightGBM, SHAP |
| Time-series | Statsmodels, Prophet |
| Deep Learning | TensorFlow / Keras |
| Notebook | Jupyter, nbformat |

---

## 👤 Author

Assessment submitted for **PM Accelerator — Data Scientist / Analyst Role**

---

## 📄 License

Open-source under the [MIT License](LICENSE).
