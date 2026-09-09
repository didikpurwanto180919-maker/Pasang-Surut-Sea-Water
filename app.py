import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Hindari GUI blocking
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Prediksi Water Intake PLTGU Grati",
    layout="wide"
)

st.title("🌊 Realtime Sea Water Intake Level - PLTGU Grati")
st.write("Sistem Monitoring & Prediksi Ketinggian Air Laut berbasis Machine Learning")

# Parameter Operasional
CRITICAL_LOW_LEVEL = 0.5
CRITICAL_HIGH_LEVEL = 2.8

# ==========================================
# 2. MODEL MACHINE LEARNING (CACHED)
# ==========================================
@st.cache_resource
def train_model():
    """Melatih model sekali saja dan menyimpannya di cache Streamlit"""
    np.random.seed(42)
    days = 180
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(hours=i) for i in range(days * 24)]
    
    df = pd.DataFrame({"datetime": dates})
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    
    t = np.arange(len(df))
    harmonic_k1 = 0.9 * np.sin(2 * np.pi * t / 23.93)
    harmonic_o1 = 0.6 * np.sin(2 * np.pi * t / 25.82 + 0.5)
    harmonic_m2 = 0.3 * np.cos(2 * np.pi * t / 12.42)
    msl_base = 1.5
    
    df["astronomical_tide"] = msl_base + harmonic_k1 + harmonic_o1 + harmonic_m2
    df["pressure_hpa"] = 1010 + 5 * np.sin(2 * np.pi * t / (24 * 7)) + np.random.normal(0, 1.5, len(df))
    df["wind_speed_m_s"] = np.abs(4 + 3 * np.cos(2 * np.pi * t / 24) + np.random.normal(0, 2, len(df)))
    df["wind_direction_deg"] = (45 + np.random.normal(0, 30, len(df))) % 360
    
    surge_pressure = (1013.25 - df["pressure_hpa"]) * 0.01
    surge_wind = (df["wind_speed_m_s"] ** 1.8) * 0.003 * np.cos(np.radians(df["wind_direction_deg"] - 45))
    df["actual_water_level"] = df["astronomical_tide"] + surge_pressure + surge_wind + np.random.normal(0, 0.03, len(df))
    
    features = ["hour", "dayofyear", "astronomical_tide", "pressure_hpa", "wind_speed_m_s", "wind_direction_deg"]
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    model.fit(df[features], df["actual_water_level"])
    
    return model

model = train_model()

# ==========================================
# 3. FUNGSI UTILITY & API
# ==========================================
def fetch_weather():
    url = "https://api.open-meteo.com/v1/forecast?latitude=-7.644317&longitude=113.027350&current=surface_pressure,wind_speed_10m,wind_direction_10m&timezone=Asia%2FJakarta"
    try:
        res = requests.get(url, timeout=5).json().get("current", {})
        return {
            "pressure_hpa": res.get("surface_pressure", 1011.0),
            "wind_speed_m_s": res.get("wind_speed_10m", 3.5) / 3.6,
            "wind_direction_deg": res.get("wind_direction_10m", 60.0)
        }
    except Exception:
        return {"pressure_hpa": 1011.0, "wind_speed_m_s": 3.0, "wind_direction_deg": 45.0}

def compute_tide(dt):
    t_hours = dt.timestamp() / 3600.0
    return 1.5 + 0.9*np.sin(2*np.pi*t_hours/23.93) + 0.6*np.sin(2*np.pi*t_hours/25.82 + 0.5) + 0.3*np.cos(2*np.pi*t_hours/12.42)

# ==========================================
# 4. STREAMLIT UI & INTERACTION
# ==========================================
now = datetime.now()
weather = fetch_weather()
astro_tide = compute_tide(now)

input_data = pd.DataFrame([{
    "hour": now.hour,
    "dayofyear": now.timetuple().tm_yday,
    "astronomical_tide": astro_tide,
    "pressure_hpa": weather["pressure_hpa"],
    "wind_speed_m_s": weather["wind_speed_m_s"],
    "wind_direction_deg": weather["wind_direction_deg"]
}])

pred_level = model.predict(input_data)[0]

# Display Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prediksi Level Water Intake", f"{pred_level:.2f} m")
col2.metric("Pasang Surut Astronomis", f"{astro_tide:.2f} m")
col3.metric("Tekanan Udara", f"{weather['pressure_hpa']:.1f} hPa")
col4.metric("Kecepatan Angin", f"{weather['wind_speed_m_s']:.1f} m/s")

# Status Warning
if pred_level <= CRITICAL_LOW_LEVEL:
    st.error("⚠️ BAHAJA: CRITICAL LOW LEVEL (Potensi Kavitasi Pompa CWP)")
elif pred_level >= CRITICAL_HIGH_LEVEL:
    st.warning("⚠️ PERINGATAN: CRITICAL HIGH LEVEL (Potensi Overflow Channel)")
else:
    st.success("✅ Status Operasional Intake: NORMAL")

# Plot Visualisasi dengan st.pyplot
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(["Level Prediksi ML"], [pred_level], color='skyblue', width=0.4)
ax.axhline(CRITICAL_HIGH_LEVEL, color='red', linestyle='--', label='Batas Atas Kritis (2.8m)')
ax.axhline(CRITICAL_LOW_LEVEL, color='orange', linestyle='--', label='Batas Bawah Kritis (0.5m)')
ax.set_ylabel("Ketinggian Air (Meter)")
ax.set_ylim(0, 3.5)
ax.legend()
ax.grid(axis='y', linestyle=':', alpha=0.7)

st.pyplot(fig) # Menggunakan st.pyplot(), bukan plt.show()
