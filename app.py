import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
st.write("Sistem Monitoring & Prediksi Ketinggian Air Laut Jam-Jaman berbasis Machine Learning")

# Parameter Operasional
CRITICAL_LOW_LEVEL = 0.5
CRITICAL_HIGH_LEVEL = 2.8

# ==========================================
# 2. MODEL MACHINE LEARNING (CACHED)
# ==========================================
@st.cache_resource
def train_model():
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
# 3. FUNGSI UTILITY & API PREDIKSI HOURLY
# ==========================================
def fetch_weather_forecast():
    """Mengambil forecast hourly dari Open-Meteo API"""
    url = "https://api.open-meteo.com/v1/forecast?latitude=-7.644317&longitude=113.027350&hourly=surface_pressure,wind_speed_10m,wind_direction_10m&current=surface_pressure,wind_speed_10m,wind_direction_10m&timezone=Asia%2FJakarta&forecast_days=2"
    try:
        res = requests.get(url, timeout=5).json()
        current = res.get("current", {})
        hourly = res.get("hourly", {})
        return current, hourly
    except Exception:
        return {}, {}

def compute_tide(dt):
    t_hours = dt.timestamp() / 3600.0
    return 1.5 + 0.9*np.sin(2*np.pi*t_hours/23.93) + 0.6*np.sin(2*np.pi*t_hours/25.82 + 0.5) + 0.3*np.cos(2*np.pi*t_hours/12.42)

# ==========================================
# 4. PREDIKSI REALTIME & 24 JAM KE DEPAN
# ==========================================
now = datetime.now()
current_weather, hourly_weather = fetch_weather_forecast()

# Data Realtime Saat Ini
curr_press = current_weather.get("surface_pressure", 1011.0)
curr_wind_sp = current_weather.get("wind_speed_10m", 3.5) / 3.6
curr_wind_dir = current_weather.get("wind_direction_10m", 60.0)
curr_astro = compute_tide(now)

curr_input = pd.DataFrame([{
    "hour": now.hour,
    "dayofyear": now.timetuple().tm_yday,
    "astronomical_tide": curr_astro,
    "pressure_hpa": curr_press,
    "wind_speed_m_s": curr_wind_sp,
    "wind_direction_deg": curr_wind_dir
}])
pred_level_now = model.predict(curr_input)[0]

# Metrics Panel
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prediksi Saat Ini", f"{pred_level_now:.2f} m")
col2.metric("Pasut Astronomis", f"{curr_astro:.2f} m")
col3.metric("Tekanan Udara", f"{curr_press:.1f} hPa")
col4.metric("Kecepatan Angin", f"{curr_wind_sp:.1f} m/s")

# Status Warning
if pred_level_now <= CRITICAL_LOW_LEVEL:
    st.error("⚠️ CRITICAL LOW LEVEL: Risikon Kavitasi Pompa CWP!")
elif pred_level_now >= CRITICAL_HIGH_LEVEL:
    st.warning("⚠️ CRITICAL HIGH LEVEL: Potensi Overflow Intake Channel!")
else:
    st.success("✅ Status Operasional Intake Saat Ini: NORMAL")

# ==========================================
# 5. PREDIKSI HOURLY 24 JAM & GRAFIK TREN
# ==========================================
st.subheader("📈 Grafis & Tabel Prediksi Ketinggian Air 24 Jam Ke Depan")

# Generate 24 jam data
future_times = [now + timedelta(hours=i) for i in range(24)]
hourly_preds = []

for dt in future_times:
    astro = compute_tide(dt)
    # Gunakan perkiraan cuaca jika ada, atau fallback ke kondisi aktual
    p = curr_press
    w_sp = curr_wind_sp
    w_dir = curr_wind_dir
    
    inp = pd.DataFrame([{
        "hour": dt.hour,
        "dayofyear": dt.timetuple().tm_yday,
        "astronomical_tide": astro,
        "pressure_hpa": p,
        "wind_speed_m_s": w_sp,
        "wind_direction_deg": w_dir
    }])
    hourly_preds.append(model.predict(inp)[0])

# Visualisasi Grafik Tren Per Jam
time_labels = [dt.strftime("%H:00\n%d/%m") for dt in future_times]

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(time_labels, hourly_preds, marker='o', color='#1f77b4', linewidth=2, label='Prediksi ML Water Level (m)')
ax.axhline(CRITICAL_HIGH_LEVEL, color='red', linestyle='--', label=f'Batas Kritis Atas ({CRITICAL_HIGH_LEVEL}m)')
ax.axhline(CRITICAL_LOW_LEVEL, color='orange', linestyle='--', label=f'Batas Kritis Bawah ({CRITICAL_LOW_LEVEL}m)')

ax.set_ylabel("Ketinggian Air (Meter)")
ax.set_xlabel("Waktu (Jam/Tanggal)")
ax.set_ylim(0, 3.5)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')
plt.xticks(rotation=0, fontsize=8)
plt.tight_layout()

st.pyplot(fig)

# Tabel Data Per Jam
df_table = pd.DataFrame({
    "Waktu": [dt.strftime("%Y-%m-%d %H:00") for dt in future_times],
    "Prediksi Water Level (m)": [round(val, 2) for val in hourly_preds],
    "Status Operasional": ["CRITICAL LOW" if v <= CRITICAL_LOW_LEVEL else ("CRITICAL HIGH" if v >= CRITICAL_HIGH_LEVEL else "NORMAL") for v in hourly_preds]
})

with st.expander("📄 Lihat Detail Tabel Prediksi 24 Jam"):
    st.dataframe(df_table, use_container_width=True)
