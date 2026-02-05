# Trip Price Prediction — ML, Unsupervised Learning & Production Deployment

End-to-end machine learning project for predicting ride prices from structured tabular data, including:

- Leakage-safe evaluation with route-based grouping
- Supervised regression modeling (**XGBoost as final model**)
- Unsupervised learning (**PCA + clustering + anomaly detection**)
- Deployment using **FastAPI + Streamlit + Hugging Face Spaces**

---

## Project Overview
Ride pricing is a real-world regression problem influenced by multiple factors such as distance, route, demand signals (surge), time-of-day, and weather conditions.

This project builds a production-ready pipeline that predicts ride prices while ensuring realistic evaluation and preventing data leakage through group-aware validation.

---

## Dataset
The dataset contains ~100,000 ride records.

Each row represents a ride with:
- Trip attributes (distance, route)
- Service attributes (cab type, name)
- Pricing signals (surge multiplier)
- Weather conditions (temperature, humidity, pressure, wind, clouds)

Target variable:
- **price**

---

## Features

### Core Features
- **Trip:** `distance`, `hour`, `route`
- **Service:** `cab_type`, `name`
- **Pricing:** `surge_multiplier`
- **Weather:** `temp`, `humidity`, `pressure`, `clouds`, `wind`
- **Time:** `month`, `day`

---

## Preprocessing
A unified preprocessing pipeline was used for all models:

- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder(handle_unknown="ignore")`
- All transformations wrapped inside a single `ColumnTransformer`

This ensures:
- consistent transformations across all models
- no leakage during train/test evaluation
- safe reuse during inference

Saved artifact:
- `models/preprocessor.pkl`

---

## Supervised Learning

### Models Trained
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (**final**)
- Neural Network (ANN) — Keras

---

## Validation Strategy
To avoid spatial leakage (same route appearing in train and test), evaluation was performed using a **group-aware split**:

- `route = source + destination`
- Train/test split ensures routes do not overlap
- All models were evaluated using the same split and metrics

This simulates a realistic scenario:
> predicting prices for unseen routes.

---

## Results

| Model | MAE | RMSE |
|------|-----:|-----:|
| Linear Regression (Baseline) | 2.229 | 3.228 |
| **XGBoost (Final)** | **1.199** | **1.840** |
| Gradient Boosting | 1.284 | 1.942 |
| Neural Network (ANN) | 1.237 | - |
| Random Forest | 2.805 | 3.854 |

### Final Model
**XGBoost** was selected for deployment due to:
- best MAE / RMSE
- strong generalization
- fast inference
- robust performance on structured tabular data

Saved artifact:
- `models/best_xgb.pkl`

---

## Unsupervised Learning
The project also includes an unsupervised learning module to discover ride behavior patterns and detect anomalies.

### PCA Visualization
PCA was applied after preprocessing to:
- reduce dimensionality
- visualize ride distribution
- support clustering interpretability

### KMeans (Ride Pattern Clustering)
KMeans clustering was used to group rides into interpretable patterns, such as:
- short frequent rides
- long high-price rides
- peak-hour demand rides
- surge-heavy clusters

Cluster profiling was performed using:
- avg price
- avg distance
- surge rate
- peak hour
- route frequency

### DBSCAN (Anomaly Detection)
DBSCAN was used for anomaly detection to identify:
- extreme surge events
- rare route behavior
- pricing outliers inconsistent with distance

---

## Deployment
This project was deployed using:

- **FastAPI** for model inference API
- **Streamlit** for interactive UI
- **Hugging Face Spaces** for hosting

### Deployment Artifacts
- `models/best_xgb.pkl`
- `models/preprocessor.pkl`

---

## Project Structure

```text
Trip-Price-Prediction/
│── README.md
│── requirements.txt
│── Dockerfile
│── .dockerignore
│── space.yaml
│
│── app.py                  # FastAPI API
│── streamlit_app.py        # Streamlit UI
│
│── models/
│   ├── best_xgb.pkl
│   └── preprocessor.pkl
│
│── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Supervised.ipynb
│   └── 03_Unsupervised.ipynb
│
└── Data/
    └── sample.csv

---

## How to Run Locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
### 2) Run FastAPI
uvicorn app:app --reload
### 3) Run Streamlit
streamlit run streamlit_app.py


🌐 Streamlit App (Hugging Face Spaces)

👉 https://huggingface.co/spaces/abdelrhman111/Trip_PricePrediction_Streamlit

