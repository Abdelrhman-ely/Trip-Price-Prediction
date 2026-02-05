# 🚖 Trip Price Prediction — ML, Unsupervised Learning & Deployment

End-to-end ML project for predicting ride prices using **XGBoost**, **unsupervised learning**, and **production deployment** on Hugging Face Spaces.

---

## 🎯 Overview

Production-ready ML system built with **~100,000 ride records** featuring:
- **Supervised Learning**: XGBoost regression (MAE: 1.199)
- **Unsupervised Learning**: KMeans clustering + DBSCAN anomaly detection
- **Group-Aware Validation**: No route leakage between train/test
- **Full Deployment**: FastAPI + Streamlit on Hugging Face Spaces

---

## 🌐 Live Demo
👉 **[https://huggingface.co/spaces/abdelrhman111/Trip_PricePrediction_Streamlit](https://huggingface.co/spaces/abdelrhman111/Trip_PricePrediction_Streamlit)**

---

## 📊 Dataset & Features

**~100,000 ride records**  
**Target:** `price`

### Feature Categories
| Category | Features |
|----------|----------|
| **Trip** | `distance`, `hour`, `source`, `destination`, `route` |
| **Vehicle** | `cab_type`, `name` |
| **Pricing** | `surge_multiplier` |
| **Weather** | `temp`, `humidity`, `pressure`, `clouds`, `wind` |
| **Time** | `month`, `day` |

---

## 🛠 Project Pipeline

1. **Preprocessing**: Unified pipeline with `StandardScaler` + `OneHotEncoder` wrapped in `ColumnTransformer`
2. **Feature Engineering**: Created `route = source + destination`, cyclical time features
3. **Supervised ML**: Trained 5 models → Selected **XGBoost**
4. **Group-Aware Validation**: Route-based split (no leakage)
5. **Unsupervised Learning**: KMeans clustering + DBSCAN anomaly detection
6. **Deployment**: FastAPI backend + Streamlit UI on Hugging Face

---

## 🤖 Supervised Learning Results

### Model Comparison

| Model | MAE ↓ | RMSE | Status |
|-------|------:|-----:|:------:|
| **XGBoost** | **1.199** | **1.840** | ✅ **Deployed** |
| Neural Network (ANN) | 1.237 | - | 🧪 Experimental |
| Gradient Boosting | 1.284 | 1.942 | - |
| Linear Regression | 2.229 | 3.228 | 📊 Baseline |
| Random Forest | 2.805 | 3.854 | - |

### Why XGBoost?
- **Lowest MAE & RMSE**
- **Strong generalization** on unseen routes
- **Robust** to outliers and missing values
- **Fast inference** for production

### Validation Strategy
**Group-aware split** based on `route`:
- Prevents same route in train & test sets
- Ensures realistic evaluation on unseen routes
- Avoids spatial data leakage

---

## 🔍 Unsupervised Learning Insights

### KMeans Clustering (Mobility Patterns)

**Cluster Profiling** analyzed:
- Ride count per cluster
- Average & median price
- Average distance
- Surge rate (% rides with surge > 1)
- Peak hour per cluster

**Discovered Patterns:**
- **Budget Rides**: Short distance, low surge, economy cabs
- **Rush Hour Premium**: High surge during peak times
- **Long Distance**: Airport/intercity trips
- **Luxury Segment**: Premium cabs, specific high-value routes

**Business Value:**
- Targeted pricing per mobility segment
- Demand forecasting based on cluster behavior
- Fleet optimization & allocation

---

### 🚨 DBSCAN Anomaly Detection

**Parameters:** `eps=1.0`, `min_samples=14`

**Features Used:**
- `distance`, `price`, `price_per_km`, `surge_multiplier`
- `hour_sin`, `hour_cos` (cyclical encoding)
- `route_freq` (route popularity)

**Detected Anomalies:**
- Unusually high `price_per_km` for given distance
- Extreme surge multipliers in off-peak hours
- Rare routes with inconsistent pricing

**Applications:**
- **Fraud Detection**: Flag suspicious pricing
- **Dynamic Pricing**: Refine rates for outlier routes
- **Quality Control**: Monitor pricing consistency

---

## 📦 Deployment Artifacts
```
models/
├── best_xgb.pkl          # Trained XGBoost model
├── preprocessor.pkl      # Preprocessing pipeline
```

---

## 💻 Run Locally

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1️⃣ Run FastAPI (Backend)
```bash
uvicorn api:app --reload
```
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### 2️⃣ Run Streamlit (Frontend)
```bash
streamlit run streamlit_app.py
```
- App: `http://localhost:8501`

---

## 📡 API Usage

### Base URL
```
Production: https://abdelrhman111-trip-priceprediction.hf.space
Local: http://localhost:8000
```

### Endpoints

#### 1. Health Check
```bash
GET /
```

**Response:**
```json
{
  "status": "API is running"
}
```

---

#### 2. Predict Price
```bash
POST /predict
```

**Request Body:**
```json
{
  "distance": 5.0,
  "cab_type": "UberX",
  "source": "Back Bay",
  "destination": "South Station",
  "surge_multiplier": 1.0,
  "name": "Uber",
  "hour": 14,
  "temp": 20.0,
  "clouds": 20.0,
  "pressure": 1013.0,
  "humidity": 50.0,
  "wind": 3.0,
  "month": 2,
  "day": 5
}
```

**Response:**
```json
{
  "Predicted Price": 18.74
}
```


**⭐ If you found this project useful, please star the repo!**
