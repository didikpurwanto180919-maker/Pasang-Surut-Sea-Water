import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prediksi Pasut Pasuruan", layout="wide")

st.title("🌊 Prediksi vs Aktual Level Air Laut")
st.subheader("Lokasi: Pasuruan (S 07° 38.659' E 113° 01.641')")

# 1. AMBIL / SIMULASI DATA
@st.cache_data
def load_data():
    time_range = pd.date_range(start="2026-08-01", periods=1000, freq="15min")
    t = np.arange(len(time_range))
    
    # Komponen Pasut M2 & S2 + Noise Cuaca
    m2_tide = 1.2 * np.sin(2 * np.pi * t / 49.6)
    s2_tide = 0.5 * np.sin(2 * np.pi * t / 48.0)
    weather_noise = np.random.normal(0, 0.08, len(t))
    
    water_level = 2.0 + m2_tide + s2_tide + weather_noise
    return pd.DataFrame({'water_level': water_level}, index=time_range)

df = load_data()

# 2. PREPROCESSING
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['water_level']])

LOOKBACK = 24

def create_features(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

X, y = create_features(scaled_data, LOOKBACK)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. TRAINING MODEL (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. PREDIKSI
predictions = model.predict(X_test)

predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 5. VISUALISASI STREAMLIT
test_timestamps = df.index[train_size + LOOKBACK:]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_timestamps, y_test_actual, label="Aktual (BIG)", color="blue", linewidth=1.5)
ax.plot(test_timestamps, predictions_actual, label="Prediksi ML (Random Forest)", color="red", linestyle="--", linewidth=1.5)
ax.set_ylabel("Tinggi Muka Air (Meter)")
ax.set_xlabel("Waktu")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)

st.pyplot(fig)
