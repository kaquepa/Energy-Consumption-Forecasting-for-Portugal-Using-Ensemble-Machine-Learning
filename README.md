# Energy Consumption Forecasting for Portugal Using Ensemble Machine Learning

> Production ready system for daily energy consumption forecasting in Portugal, achieving **MAPE 2.45%**, with automated pipeline, REST API and interactive dashboard.

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Model Performance](https://img.shields.io/badge/R²-0.891-brightgreen.svg)
![MAPE](https://img.shields.io/badge/MAPE-2.45%25-success.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-FF4B4B.svg)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Dashboard](#-dashboard)
- [Daily Pipeline](#-daily-pipeline)
- [Technologies](#-technologies)
- [Data Sources](#-data-sources)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)

---

## Overview 

**Energy Consumption Forecasting for Portugal** is a production-ready machine learning system that predicts Portugal's electricity demand achieving **2.45% MAPE** (89.1% variance explained, R^2=0.891). The system combines competition-based model selection with real-time weather forecasts to deliver precise predictions up to 7 days ahead.

Built for operational deployment, the system includes automated data pipelines, REST API endpoints, and an interactive dashboard for real-time monitoring and analysis.

---

## Key Highlights

- **World-class performance** in energy forecasting (MAPE 2.45% - "Excellent" tier)
- **7-day multi-horizon forecasting** with confidence intervals (68%, 95%)
- **Portuguese-specific features**: 13 national holidays + bridge day detection
- **Weather-aware predictions** via Open-Meteo API integration
- **Professional dashboard** built with Streamlit for real-time monitoring
- **Production REST API** with FastAPI and smart caching (82% latency reduction)
- **Automated daily pipeline** for continuous operation
- **15+ years historical data** (2010-2026, 5849 days)
- **Competition-based selection**: Best of 3 algorithms per horizon
- **SHAP guided features**: Interpretable, principled selection

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-horizon Forecasting** | Predict 1-7 days ahead with individual confidence levels |
| **Model Competition** | 3 models per horizon (RandomForest, LightGBM, XGBoost) - best wins |
| **Weather Integration** | Real-time weather forecasts for enhanced accuracy |
| **Confidence Intervals** | 68% and 95% prediction intervals based on MAE |
| **Smart Feature Selection** | SHAP based selection of top 30 from 55 candidates |
| **Portuguese Holidays** | Complete national holiday detection + bridge days |
| **Professional Dashboard** | Real-time visualization with Streamlit |
| **REST API** | FastAPI endpoints with TTL caching (5min) |
| **Automated Pipeline** | Daily data collection, training, and prediction |

### Prediction Accuracy by Horizon

| Horizon | RMSE (GWh) | R^2 | MAE (GWh) | MAPE | Confidence |
|---------|------------|-----|-----------|------|------------|
| **Day +1** | **5.49** | **0.891** | **3.46** | **2.45%** | HIGH  |
| **Day +2** | 6.61 | 0.842 | 4.60 | 3.21% | HIGH |
| **Day +3** | 7.12 | 0.817 | 5.14 | 3.59% | MEDIUM |
| **Day +4** | 7.64 | 0.789 | 5.47 | 3.80% | MEDIUM |
| **Day +5** | 7.61 | 0.790 | 5.43 | 3.77% | MEDIUM |
| **Day +6** | 8.01 | 0.768 | 5.72 | 3.95% | MEDIUM |
| **Day +7** | 8.18 | 0.759 | 5.75 | 3.95% | MEDIUM |
| **Average** | **7.24** | **0.808** | **5.08** | **3.53%** | - |

> **Benchmark (Lewis, 1982)**: MAPE < 3% = Excellent  | 3-5% = Good | 5-10% = Reasonable | >10% = Poor

---

## Installation

### Prerequisites

- Python 3.12+
- pip or conda package manager
- Git

### Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/kaquepa/Energy-Consumption-Forecasting-for-Portugal-Using-Ensemble-Machine-Learning.git
```


**4. Initialize the system**
```bash
# Manual execution
bash ./code/scripts/daily_pipeline.sh

#Start the Dashboard
bash ./code/scripts/api_start.sh

# Check last prediction timestamp
cat data/predictions/latest_prediction.json | grep generated_at

# Check model registry
cat code/trainer/models/registry.json

# Schedule with cron (daily at 2 AM)
0 2 * * * /path/to/project/code/scripts/daily_pipeline.sh >> /path/to/logs/cron.log 2>&1
```

---

##  Quick Start
**Access:**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### Make Your First Prediction
```bash
# Get tomorrow's forecast
curl http://localhost:8000/energy/predict-next

# Get 7-day forecast
curl http://localhost:8000/energy/forecast/7

# Check system health
curl http://localhost:8000/health
```
---

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                          │
│ REN API (Consumption, Production) + Weather API(Open-Meteo) │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                         │
│  - 30 engineered features via SHAP selection                │
│  - Time features (weekday, month, holidays)                 │
│  - Lag features (1-30 days)                                 │
│  - Rolling statistics (3, 7, 14, 30 days)                   │
│  - Weather features + interactions                          │
│  - Portuguese national holidays + bridge days               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING                            │
│  1. Model Competition (per horizon):                        │
│     - RandomForest                                          │
│     - LightGBM (wins 4/7)                                   │
│     - XGBoost (wins 2/7)                                    │
│  2. Best model selected based on validation RMSE            │
│  3. 7 independent models (direct forecasting)               │
│  4. Cross-validation + performance tracking                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION LAYER                          │
│  FastAPI REST API + Streamlit Dashboard                     │
│  - Multi-horizon predictions (1-7 days)                     │
│  - Confidence intervals (68%, 95%)                          │
│  - Weather integration                                      │
│  - Smart caching (TTL 5min, 82% latency reduction)          │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Collection**: Download incremental data from REN API + weather forecasts ($\approx$ 30s)
2. **Processing**: Transform raw data + create 30 engineered features ($\approx$ 10s)
3. **Training**: Competition based selection per horizon (weekly retrain, $\approx$ 1min)
4. **Prediction**: Generate 7-day forecast with confidence intervals (<1s)
5. **Serving**: API endpoints + Dashboard visualization (real-time)

---

##  Model Performance

### Overall Metrics

| Metric | Day +1 | Day +2 | Day +3 | Day +7 | System Average |
|--------|--------|--------|--------|--------|----------------|
| **RMSE (GWh)** | 5.49 | 6.61 | 7.12 | 8.18 | 7.24 |
| **R^2 Score** | 0.891 | 0.842 | 0.817 | 0.759 | 0.808 |
| **MAE (GWh)** | 3.46 | 4.60 | 5.14 | 5.75 | 5.08 |
| **MAPE (%)** | **2.45** | 3.21 | 3.59 | 3.95 | 3.53 |

**Interpretation:**
- **MAPE 2.45%** = Excellent forecast quality (Lewis, 1982)
- **R^2 0.891** = Model explains 89.1% of consumption variance
- **MAE 3.46 GWh** = Average error of $\approx$ 3.5 GWh ( $\approx$ 2.5% of typical 137.6 GWh daily consumption)
- **52% error reduction** vs. 7-day moving average baseline (5.1% $\rightarrow$ 2.45%)

**Performance Classification (Lewis, 1982):**
- MAPE < 3% = Excellent 
- MAPE 3-5% = Good 
- MAPE 5-10% = Reasonable
- MAPE > 10% = Poor

### Confidence Intervals

Prediction intervals based on Mean Absolute Error (MAE):

- **68% interval**: $\pm$ 1.0 × MAE (e.g., Day+1: $\pm$ 3.46 GWh)
- **95% interval**: $\pm$ 1.96 × MAE (e.g., Day+1: $\pm$ 6.78 GWh)

**Example Interpretation:**

Prediction: 169.75 GWh (Day +1)
68% interval: [166.29, 173.21] GWh  $\rightarrow$ 68% probability actual value falls here
95% interval: [162.97, 176.53] GWh  $\rightarrow$  95% probability actual value falls here

### Competition Winners by Horizon

| Horizon | Winner | RMSE | Runner up | **$\Delta$** RMSE |
|---------|--------|------|-----------|--------|
| h=1 | **LightGBM** | 5.49 | XGBoost | +0.12 |
| h=2 | **XGBoost** | 6.61 | LightGBM | +0.08 |
| h=3 | **LightGBM** | 7.12 | RandomForest | +0.23 |
| h=4 | **LightGBM** | 7.64 | XGBoost | +0.11 |
| h=5 | **XGBoost** | 7.61 | LightGBM | +0.05 |
| h=6 | **LightGBM** | 8.01 | XGBoost | +0.09 |
| h=7 | **RandomForest** | 8.18 | LightGBM | +0.14 |

**Key Insights:**
- LightGBM dominates short to medium range (4/7 horizons)
- XGBoost excels at mid-range forecasting (2/7 horizons)
- RandomForest wins long-range prediction (h=7)
- No single algorithm optimal for all horizons $\rightarrow$ validates competition approach

### Model Architecture

**Direct Multi-Horizon Forecasting:**
- Independent model for each horizon (h=1 to h=7)
- No error propagation between predictions
- Industry standard approach (80% of GEFCom2014 winners)
- Each model optimized for specific forecast distance

**Why Direct vs. Recursive?**
- **Direct**: Independent predictions $\rightarrow$ no error compounding
- **Recursive**: Uses previous predictions as inputs $\rightarrow$ errors propagate
- **Proven**: Direct approach superior for energy forecasting (Hong et al., 2016)

### Feature Importance (Top 10)

Based on SHAP values for horizon h=1:

1. **consumption_lag_1** (12.8) - Yesterday's consumption
2. **consumption_lag_2** (8.4) - Day before yesterday
3. **rolling_mean_7** (7.9) - Weekly momentum
4. **consumption_lag_7** (6.2) - Same weekday last week
5. **rolling_mean_30** (5.8) - Monthly trend
6. **temperature_mean** (4.7) - Primary weather driver
7. **rolling_std_7** (3.9) - Weekly volatility
8. **trend_7days** (3.2) - Directional change detector
9. **temp_x_humidity** (2.8) - Compound weather effect
10. **is_weekend** (2.1) - Business activity marker

**Feature Category Distribution:**
- Lag features: 40%
- Rolling statistics: 30%
- Weather: 18%
- Temporal: 7%
- Interactions: 5%

### Holiday Impact

**Portuguese-Specific Features:**
- 13 National Holidays (9 fixed + 4 variable)
- Bridge day detection (days between holidays and weekends)

**Performance Impact:**
- Holidays reduce consumption by $\approx$15-20%
- Bridge days reduce by $\approx$ 8-12%
- Adding holiday features improved MAPE by **28%** (5.8% $\rightarrow$ 4.2%) on holiday periods

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Energy Endpoints

#### Get Next Day Prediction
```bash
GET /energy/predict-next
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "date": "2026-01-09",
    "predicted": 169.75,
    "confidence_interval": {
      "lower_68": 166.29,
      "upper_68": 173.21,
      "lower_95": 162.97,
      "upper_95": 176.53
    },
    "weather": {
      "temperature": 10.0,
      "radiation": 8.4,
      "humidity": 86,
      "rain": 0.0,
      "wind": 11.0
    },
    "metrics": {
      "RMSE": 5.49,
      "MAE": 3.46,
      "MAPE": 2.45,
      "R2": 0.891
    }
  },
  "generated_at": "2026-01-08T23:40:31.241591"
}
```

#### Get Multi-Day Forecast
```bash
GET /energy/forecast/{days}
```

**Parameters:**
- `days` (int): Number of days to forecast (1-7)

**Example:**
```bash
curl http://localhost:8000/energy/forecast/7
```

**Response:**
```json
{
  "status": "success",
  "last_known": {
    "date": "2026-01-08",
    "consumption": 175.2
  },
  "predictions": {
    "day_1": {
      "date": "2026-01-09",
      "predicted": 169.75,
      "weather": {...},
      "metrics": {...}
    },
    "day_2": {...},
    ...
    "day_7": {...}
  },
  "generated_at": "2026-01-08T23:40:31.241591"
}
```

### Weather Endpoints

#### Get Current Weather
```bash
GET /weather/current
```

#### Get Weather Forecast
```bash
GET /weather/forecast/{days}
```

**Parameters:**
- `days` (int): Number of days (1-16)

### System Endpoints

#### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-08T23:40:31.241591",
  "version": "1.0.0"
}
```

#### Model Information
```bash
GET /model/info
```

**Response:**
```json
{
  "models": {
    "h1": {
      "algorithm": "LightGBM",
      "RMSE": 5.49,
      "MAPE": 2.45,
      "R2": 0.891,
      "trained_at": "2026-01-06"
    },
    ...
  },
  "feature_count": 30,
  "dataset_size": 5849
}
```

### API Features

- **Smart Caching**: TTL-based caching (5min) reduces latency by 82%
- **Error Handling**: Comprehensive error responses with status codes
- **Rate Limiting**: Prevents API abuse
- **CORS Support**: Enabled for cross origin requests
- **Interactive Docs**: Swagger UI at `/docs`

---

## Dashboard

The Streamlit dashboard provides real-time visualization and monitoring of the forecasting system.

### Pages

**1. Home**
- 7-day forecast table with predicted values
- Historical consumption chart (90 days)
- Forecast overlay with confidence bands
- Today's prediction with change indicator
- Real-time update timestamp

![Dashboard Home](images/home.png)

**2. Weather**
- Current weather conditions for Central Portugal
- 7-day weather forecast (horizontal cards)
- Temperature, radiation, humidity, rain, wind
- Weather icons and intuitive visualization

![Weather Dashboard](images/weather.png)

**3. EDA (Exploratory Data Analysis)**
- Historical trends (15 years, 2010-2026)
- Outlier detection with statistical bounds
- Weekly and seasonal pattern decomposition
- Interactive zoom and pan capabilities

![EDA Dashboard - History](images/history.png)
![EDA Dashboard - Outliers](images/outlier.png)
![EDA Dashboard - Patterns](images/week.png)

**4. Performance**
- Model metrics per horizon (RMSE, MAE, MAPE, R²)
- Error distribution histograms
- Residual analysis plots
- Feature importance rankings (SHAP)

### Access Dashboard
Navigate to: **http://localhost:8501**

### Dashboard Features

- **Auto-refresh**: Updates every 5 minutes
- **Interactive charts**: Zoom, pan, hover tooltips
- **Responsive design**: Works on desktop and tablet
- **Real-time status**: Shows last update timestamp

---

### Pipeline Stages

**1. Data Collection** (Daily: Mon-Sun)
 
Duration: $\approx$ 30 seconds
- Incremental download from REN API
- Fetch 7-day weather forecast from Open-Meteo
- Update dataset_base.csv
- Validate data quality
 

**2. Feature Engineering** (Daily: Mon-Sun)
 
Duration: $\approx$ 10 seconds
- Process new data points
- Generate 30 features per sample
- Update dataset_trainer_final.csv
- Update dataset_production_final.csv
 

**3. Model Training** (Weekly: Monday only)
 
Duration: $\approx$ 1 minutes
- Retrain all 7 horizon models
- Competition: RandomForest vs LightGBM vs XGBoost
- Validate on holdout set
- Update model registry with new metrics
- Promote best models to production
 

**4. Prediction** (Daily: Mon-Sun)

Duration: <1 second
- Generate 7-day forecast
- Compute confidence intervals
- Update latest_prediction.json
- Refresh dashboard data
 
---

##  Technologies

### Core ML Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12 | Core language |
| LightGBM | 4.1 | Gradient boosting (winner 4/7 horizons) |
| XGBoost | 2.0 | Gradient boosting (winner 2/7 horizons) |
| scikit-learn | 1.3 | RandomForest, preprocessing, metrics |
| pandas | 2.1 | Data manipulation |
| numpy | 1.25 | Numerical computing |
| SHAP | 0.43 | Feature importance & interpretability |

### API & Dashboard

| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.115.0 | REST API framework |
| Uvicorn | 0.32.0 | ASGI server |
| Streamlit | 1.40.0 | Interactive dashboard |
| Pydantic | 2.5 | Data validation |

### Data Collection

| Technology | Purpose |
|------------|---------|
| requests | HTTP client for APIs |
| REN API | Historical consumption & production data |
| Open-Meteo API | Weather forecasts |

---

## Data Sources

### REN (Redes Energéticas Nacionais)

**Type:** Official Portuguese transmission system operator

**Data:**
- Electricity consumption (national aggregate)
- Electricity production by source
- Historical data: 2010 - present

**API:**
- Endpoint: https://www.ren.pt/consumo-e-producao
- Update frequency: Hourly (aggregated to daily)
- Format: JSON
- Authentication: None (public)

**Coverage:**
- 5849 days (Jan 1, 2010 - Jan 6, 2026)
- Completeness: 99.98% (1 missing day)
- Quality: Validated, no duplicates

### Open-Meteo

**Type:** Open-source weather API

**Data:**
- Temperature (mean, min, max)
- Solar radiation (shortwave sum)
- Relative humidity
- Precipitation
- Wind speed (10m height)

**API:**
- Endpoint: https://api.open-meteo.com/v1/forecast
- Forecast range: 16 days
- Spatial resolution: 11km × 11km
- Location: Central Portugal (39.5 $^\circ$ N, 8.0 $^\circ% W ) 
- Format: JSON
- Authentication: None (free)


---

## Roadmap

### Completed (v1.0)

- [x] Historical data collection (2010-2026, 5849 days)
- [x] Feature engineering pipeline (55 $\rightarrow$ 30 features via SHAP)
- [x] Multi-horizon forecasting (7 independent models)
- [x] Model competition (RandomForest, LightGBM, XGBoost)
- [x] Confidence intervals (68%, 95% based on MAE)
- [x] Portuguese holiday detection (13 holidays + bridge days)
- [x] REST API with FastAPI
- [x] Professional Streamlit dashboard (4 pages)
- [x] Automated daily pipeline
- [x] Weather integration (Open-Meteo API)
- [x] Smart caching (TTL 5min, 82% latency reduction)
- [x] Cross-validation framework (5-fold expanding window)
- [x] Production deployment documentation
- [x] Academic paper (IEEE format)

### In Progress (v1.1)

- [ ] Unit tests (target: 80% coverage)
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Monitoring & alerting (Prometheus or Grafana)
- [ ] API authentication (API keys)
- [ ] Database integration (PostgreSQL)

###  Planned (v2.0)

- [ ] Hourly forecasting (currently daily only)
- [ ] Regional forecasting (18 districts + 2 autonomous regions)
- [ ] Probabilistic forecasting (quantile regression)
- [ ] Concept drift detection (automated retraining triggers)
- [ ] Deep learning benchmarking (Temporal Fusion Transformers, N-BEATS)
- [ ] Multi-country expansion (Spain, France)
- [ ] Renewable generation forecasting (wind, solar, hydro)
- [ ] Electricity price prediction

###  Future Research

- [ ] Causal inference framework (Granger causality)
- [ ] Transfer learning across countries
- [ ] Ensemble stacking meta-learner
- [ ] Automated hyperparameter optimization (Optuna)
- [ ] Real-time inference optimization (<100ms)
- [ ] Explainable AI dashboard (LIME, SHAP)

---

## Author

**Domingos Kaquepa Luciano Graciano**

- **Institution:** University of Aveiro, Portugal
- **Program:** Specialization Program in Machine Learning and Data Analysis
- **Email:** dkgraciano92@ua.pt
- **LinkedIn:** [Domingos Graciano](https://www.linkedin.com/in/domingos-graciano)
- **GitHub:** [@kaquepa](https://github.com/kaquepa)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **REN (Redes Energéticas Nacionais)** for providing open access to consumption data
- **Open-Meteo** for free weather API access
---

## Citation

If you use this work in your research, please cite:
```bibtex
@article{graciano2026energy,
  title={Energy Consumption Forecasting for Portugal Using Ensemble Machine Learning},
  author={Graciano, Domingos Kaquepa Luciano},
  journal={University of Aveiro},
  year={2026}
}
```

---

## Related Resources

- **Academic Paper:** [report/paper.pdf](report/paper.pdf)
- **API Documentation:** http://localhost:8000/docs (when running)
- **Dashboard:** http://localhost:8501 (when running)
- **GEFCom2014 Competition:** [IEEE Working Group on Energy Forecasting](http://www.gefcom.org/)
- **Lewis (1982) Benchmark:** CD Lewis, "Industrial and business forecasting methods"

---

<div align="center">

**If you find this project useful, please consider giving it a star!**

Made by [Domingos Graciano](https://github.com/kaquepa)

</div>