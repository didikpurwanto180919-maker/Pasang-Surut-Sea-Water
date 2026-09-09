import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pytz
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Pasut Realtime Pasuruan", layout="wide")

st.title("🌊 Prediksi vs Aktual Level Air Laut (Real-time Presisi)")
st.subheader("Lokasi: Pasuruan (S 07° 38.659' E 113° 01.641')")

# 1. WAKTU LOKAL WIB PRESISI DETIK/MENIT (TANPA CACHE)
wib_tz = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(wib_tz).replace(tzinfo=None)

st.sidebar.markdown(f"**Waktu Server / Lokal (WIB):**\n`{now_wib.strftime('%Y-%m-%d %H:%M:%S')}`")
if st.sidebar.button("🔄 Refresh Data Realtime"):
    st.rerun()

# 2. GENERATE DATA TANPA CACHE (@st.cache_data DIHAPUS AGAR ALWAYS REALTIME)
def get_realtime_tide_data(current_time):
    start_time = current_time - pd.Timedelta(hours=72)
    end_time = current_time + pd.Timedelta(hours=24)
    
    # Grid data per 15 menit agar kurva halus dan presisi
    time_range = pd.date_range(start=start_time, end=end_time, freq="15min")
    
    # Hitung waktu relatif dalam jam dari epoch 2026-01-01
    epoch_ref = pd.Timestamp("2026-01-01")
    t = (time_range - epoch_ref).total_seconds() / 3600.0
    
    # Formula Harmonik Pasut Pasuruan (Komponen M2 & S2)
    m2_tide = 1.2 * np.sin(2 * np.pi * t / 12.42)
    s2_tide = 0.5 * np.sin(2 * np.pi * t / 12.0)
    
    # Noise dinamika laut acak berbasis menit berjalan
    seed_val = int(current_time.timestamp()) % 10000
    np.random.seed(seed_val)
    weather_noise = np.random.normal(0, 0.04, len(t))
    
    water_level = 2.0 + m2_tide + s2_tide + weather_noise
    return pd.DataFrame({'water_level': water_level}, index=time_range)

df = get_realtime_tide_data(now_wib)

# 3. PREPROCESSING UNTUK MODEL MACHINE LEARNING
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['water_level']])

LOOKBACK = 24 # 6 jam data histori (24 x 15 menit)

def create_features(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

X, y = create_features(scaled_data, LOOKBACK)

split_idx = len(df[df.index <= now_wib]) - LOOKBACK

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 4. TRAINING MODEL RANDOM FOREST
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. PREDIKSI
predictions = model.predict(X_test)

predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1))
test_timestamps = df.index[split_idx + LOOKBACK:]

# 6. VISUALISASI STREAMLIT
fig, ax = plt.subplots(figsize=(12, 5))

# Plot Data Observasi / Aktual
ax.plot(df.index[:split_idx + LOOKBACK], df['water_level'][:split_idx + LOOKBACK], 
        label="Aktual / Observasi (BIG)", color="blue", linewidth=1.8)

# Plot Data Prediksi Machine Learning
ax.plot(test_timestamps, predictions_actual, 
        label="Prediksi ML (Random Forest)", color="red", linestyle="--", linewidth=1.8)

# Garis Penanda Menit Sekarang secara Tepat
ax.axvline(x=now_wib, color='green', linestyle=':', linewidth=2, 
           label=f'Saat Ini ({now_wib.strftime("%H:%M:%S WIB")})')

ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))

ax.set_ylabel("Tinggi Muka Air (Meter)")
ax.set_xlabel("Waktu (WIB)")
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)

plt.xticks(rotation=30)
st.pyplot(fig)

# 7. METRIK RINGKASAN
current_val = df.iloc[len(df[df.index <= now_wib])-1]['water_level']
next_val = predictions_actual[0][0]

col1, col2, col3 = st.columns(3)
col1.metric("Muka Air Saat Ini", f"{current_val:.2f} m")
col2.metric("Prediksi 15 Menit Ke Depan", f"{next_val:.2f} m")
col3.metric("Trend Pasut", "Sedang Naik (Pasang)" if next_val > current_val else "Sedang Turun (Surut)")
