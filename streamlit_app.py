import streamlit as st
import requests

st.set_page_config(page_title="Trip Price Prediction", layout="centered")
st.title("🚕 Trip Price Prediction")

API_BASE = "https://abdelrhman111-trip-priceprediction.hf.space"

with st.form("trip_form"):
    st.subheader("Trip Features")

    col1, col2 = st.columns(2)
    with col1:
        distance = st.number_input("distance (km)", min_value=0.01, value=5.0, step=0.1)
        surge_multiplier = st.number_input("surge_multiplier", min_value=1.0, value=1.0, step=0.05)
        hour = st.number_input("hour", min_value=0, max_value=23, value=14, step=1)
        month = st.number_input("month", min_value=1, max_value=12, value=2, step=1)
        day = st.number_input("day", min_value=1, max_value=31, value=5, step=1)

    with col2:
        temp = st.number_input("temp", value=20.0, step=0.5)
        clouds = st.number_input("clouds", value=20.0, step=1.0)
        pressure = st.number_input("pressure", value=1013.0, step=1.0)
        humidity = st.number_input("humidity", value=50.0, step=1.0)
        wind = st.number_input("wind", value=3.0, step=0.1)

    st.subheader("Categorical Fields")
    cab_type = st.text_input("cab_type", value="UberX")
    source = st.text_input("source", value="Back Bay")
    destination = st.text_input("destination", value="South Station")
    name = st.text_input("name", value="Uber")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    payload = {
        "distance": float(distance),
        "cab_type": cab_type,
        "source": source,
        "destination": destination,
        "surge_multiplier": float(surge_multiplier),
        "name": name,
        "hour": int(hour),
        "temp": float(temp),
        "clouds": float(clouds),
        "pressure": float(pressure),
        "humidity": float(humidity),
        "wind": float(wind),
        "month": int(month),
        "day": int(day),
    }

    try:
        with st.spinner("Calling API..."):
            r = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)

        if r.status_code != 200:
            st.error(f"API Error {r.status_code}")
            st.code(r.text)
        else:
            data = r.json()
            price = data.get("Predicted Price", None)
            st.success(f"Predicted Price: {price}")

    except Exception as e:
        st.error(str(e))

st.divider()
st.write("API health check:")
try:
    health = requests.get(f"{API_BASE}/", timeout=10).json()
    st.json(health)
except Exception as e:
    st.warning(f"Could not reach API: {e}")
