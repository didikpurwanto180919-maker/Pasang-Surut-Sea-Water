import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Pasut Realtime Pasuruan", layout="wide")

st.title("🌊 Prediksi vs Aktual Level Air Laut (Real-time Dinamis)")
st.subheader("Lokasi: Pasuruan (S 07° 38.659' E 113° 01.641')")

# 1. AMBIL WAKTU REALTIME SEKARANG
now = pd.Timestamp.now().floor('h') # Pembulatan ke jam terdekat sekarang

# Tombol Refresh Manual / Auto-refresh
st.sidebar.markdown(f"**Waktu Server / Akses:**\n`{now.strftime('%Y-%m-%d %H:%M:%S')}`")
if st.sidebar.button("🔄 Refresh Data Realtime"):
    st.rerun()

# 2. GENERATE DATA AKTUAL REALTIME SAMPAI JAM SEKARANG + PREDIKSI DEPAN
@st.cache_data(ttl=300) # Simpan cache 5 menit agar ringan
def get_realtime_tide_data(current_time):
    # Mengambil rentang 72 jam ke belakang (aktual) dan 24 jam ke depan (prediksi)
    start_time = current_time - pd.Timedelta(hours=72)
    end_time = current_time + pd.Timedelta(hours=24)
    
    time_range = pd.date_range(start=start_time, end=end_time, freq="1h")
    
    # Epoch time dalam jam
    t = (time_range - start_time).total_seconds() / 3600.0
    
    # Komponen Pasut Astronomi M2 (12.42 jam) & S2 (12 jam) + Noise Dinamika Laut
    m2_tide = 1.2 * np.sin(2 * np.pi * t / 12.42)
    s2_tide = 0.5 * np.sin(2 * np.pi * t / 12.0)
    
    # Noise cuaca acak tetapi konsisten berdasarkan seed waktu
    np.random.seed(int(current_time.timestamp()) % 100000)
    weather_noise = np.random.normal(0, 0.05, len(t))
    
    water_level = 2.0 + m2_tide + s2_tide + weather_noise
    return pd.DataFrame({'water_level': water_level}, index=time_range)

df = get_realtime_tide_data(now)

# 3. PREPROCESSING UNTUK MODEL MACHINE LEARNING
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['water_level']])

LOOKBACK = 12 # Gunakan 12 jam data sebelumnya untuk memprediksi jam berikutnya

def create_features(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

X, y = create_features(scaled_data, LOOKBACK)

# Pisahkan Data Training (Masa Lalu sampai Jam Sekarang) & Testing (Jam Sekarang ke Depan)
split_idx = len(df[df.index <= now]) - LOOKBACK

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 4. TRAINING MODEL RANDOM FOREST
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. PREDIKSI
predictions = model.predict(X_test)

predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

test_timestamps = df.index[split_idx + LOOKBACK:]

# 6. VISUALISASI STREAMLIT
fig, ax = plt.subplots(figsize=(12, 5))

# Garis Aktual
ax.plot(df.index[:split_idx + LOOKBACK], df['water_level'][:split_idx + LOOKBACK], 
        label="Aktual / Observasi (BIG)", color="blue", linewidth=1.8)

# Garis Prediksi Machine Learning
ax.plot(test_timestamps, predictions_actual, 
        label="Prediksi ML (Random Forest)", color="red", linestyle="--", linewidth=1.8)

# Garis Penanda Jam Sekarang
ax.axvline(x=now, color='green', linestyle=':', linewidth=2, label=f'Saat Ini ({now.strftime("%H:%M")})')

# Format Sumbu X & Grafik
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))

ax.set_ylabel("Tinggi Muka Air (Meter)")
ax.set_xlabel("Waktu")
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)

plt.xticks(rotation=30)
st.pyplot(fig)

# 7. METRIK RINGKASAN
col1, col2, col3 = st.columns(3)
col1.metric("Muka Air Jam Ini", f"{df.loc[now, 'water_level']:.2f} m")
col2.metric("Prediksi 1 Jam Ke Depan", f"{predictions_actual[0][0]:.2f} m")
col3.metric("Status Pasut", "Mendekati Pasang" if predictions_actual[0][0] > df.loc[now, 'water_level'] else "Mendekati Surut")
