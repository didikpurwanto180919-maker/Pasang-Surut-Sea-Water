import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pytz
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Pasut Probolinggo - Pasuruan", layout="wide")

st.title("🌊 Prediksi vs Aktual Level Air Laut (Sesuai Tabel BIG)")
st.subheader("Lokasi: Probolinggo / Pasuruan (S 07° 44' E 113° 12') - GMT+07.00")

# 1. PAKSA WAKTU REALTIME WIB
wib_tz = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(wib_tz).replace(tzinfo=None)

st.sidebar.markdown(f"**Waktu Realtime Saat Ini:**\n`{now_wib.strftime('%Y-%m-%d %H:%M:%S WIB')}`")
if st.sidebar.button("🔄 Refresh Data Realtime"):
    st.rerun()

# 2. TABEL AKTUR RESMI BIG (SEPTEMBER 2026: TANGGAL 1 - 30, JAM 1 - 24)
# Data diekstrak langsung dari Tabel Resmi BIG/DISHIDROS
tabel_big_sep_2026 = {
    1:  [2.4, 2.3, 2.0, 1.6, 1.2, 0.9, 0.9, 1.0, 1.2, 1.6, 1.9, 2.1, 2.2, 2.1, 1.8, 1.5, 1.2, 1.0, 0.9, 1.0, 1.3, 1.7, 2.1, 2.3],
    2:  [2.5, 2.4, 2.2, 1.8, 1.5, 1.1, 0.9, 0.9, 1.0, 1.2, 1.5, 1.8, 1.9, 2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.1, 1.3, 1.6, 1.9, 2.2],
    3:  [2.4, 2.5, 2.4, 2.1, 1.8, 1.4, 1.1, 1.0, 0.9, 1.0, 1.2, 1.4, 1.6, 1.7, 1.7, 1.6, 1.5, 1.4, 1.3, 1.3, 1.4, 1.6, 1.8, 2.1],
    4:  [2.3, 2.4, 2.3, 2.1, 1.8, 1.5, 1.2, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.6, 1.8, 1.9],
    5:  [2.1, 2.2, 2.3, 2.3, 2.3, 2.1, 1.9, 1.6, 1.3, 1.1, 0.9, 0.8, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.7, 1.7, 1.7, 1.8, 1.8, 1.8],
    6:  [1.9, 2.0, 2.1, 2.2, 2.3, 2.3, 2.2, 2.0, 1.7, 1.4, 1.1, 0.8, 0.6, 0.6, 0.7, 0.9, 1.1, 1.4, 1.7, 1.9, 1.9, 1.9, 1.9, 1.8],
    7:  [1.8, 1.8, 1.9, 2.0, 2.2, 2.3, 2.4, 2.4, 2.1, 1.8, 1.4, 1.0, 0.6, 0.4, 0.4, 0.5, 0.8, 1.2, 1.6, 1.9, 2.1, 2.1, 2.0, 1.9],
    8:  [1.7, 1.6, 1.6, 1.7, 1.9, 2.2, 2.4, 2.6, 2.5, 2.2, 1.8, 1.3, 0.8, 0.5, 0.2, 0.2, 0.5, 0.9, 1.4, 1.8, 2.2, 2.3, 2.2, 2.0],
    9:  [1.8, 1.5, 1.4, 1.4, 1.6, 1.9, 2.3, 2.6, 2.7, 2.6, 2.3, 1.8, 1.2, 0.7, 0.3, 0.1, 0.2, 0.6, 1.1, 1.6, 2.1, 2.4, 2.4, 2.2],
    10: [1.9, 1.6, 1.3, 1.2, 1.3, 1.5, 1.9, 2.4, 2.7, 2.8, 2.6, 2.2, 1.6, 1.0, 0.5, 0.2, 0.2, 0.4, 0.8, 1.4, 1.9, 2.3, 2.5, 2.4],
    11: [2.1, 1.7, 1.3, 1.1, 1.0, 1.2, 1.6, 2.0, 2.5, 2.7, 2.7, 2.5, 2.0, 1.4, 0.9, 0.4, 0.3, 0.3, 0.7, 1.2, 1.7, 2.2, 2.5, 2.5],
    12: [2.2, 1.9, 1.4, 1.1, 0.9, 0.9, 1.2, 1.6, 2.1, 2.5, 2.7, 2.6, 2.3, 1.8, 1.3, 0.8, 0.5, 0.4, 0.6, 1.0, 1.5, 2.1, 2.4, 2.5],
    13: [2.4, 2.0, 1.6, 1.2, 0.9, 0.8, 0.9, 1.3, 1.7, 2.1, 2.5, 2.6, 2.4, 2.1, 1.6, 1.1, 0.8, 0.6, 0.7, 1.0, 1.4, 1.9, 2.3, 2.5],
    14: [2.4, 2.2, 1.8, 1.4, 1.0, 0.8, 0.8, 1.0, 1.3, 1.7, 2.1, 2.3, 2.3, 2.1, 1.8, 1.4, 1.1, 0.9, 0.9, 1.1, 1.4, 1.8, 2.1, 2.4],
    15: [2.4, 2.3, 2.0, 1.6, 1.2, 0.9, 0.8, 0.9, 1.1, 1.4, 1.7, 2.0, 2.1, 2.0, 1.9, 1.6, 1.4, 1.2, 1.2, 1.3, 1.5, 1.8, 2.1, 2.3],
    16: [2.4, 2.3, 2.1, 1.8, 1.4, 1.1, 0.9, 0.9, 1.0, 1.1, 1.4, 1.6, 1.7, 1.8, 1.8, 1.7, 1.5, 1.4, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2],
    17: [2.3, 2.3, 2.1, 1.9, 1.6, 1.3, 1.1, 1.0, 1.0, 1.0, 1.1, 1.3, 1.4, 1.5, 1.5, 1.6, 1.6, 1.6, 1.6, 1.7, 1.8, 1.9, 2.1, 2.2],
    18: [2.2, 2.2, 2.1, 2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.0, 1.0, 1.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
    19: [2.2, 2.2, 2.1, 2.0, 1.9, 1.8, 1.6, 1.4, 1.3, 1.1, 1.0, 1.0, 0.9, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.1, 2.1, 2.2],
    20: [2.1, 2.1, 2.0, 2.0, 2.0, 1.9, 1.8, 1.7, 1.5, 1.3, 1.1, 1.0, 0.8, 0.7, 0.8, 0.9, 1.1, 1.4, 1.6, 1.9, 2.0, 2.1, 2.2, 2.1],
    21: [2.1, 2.0, 2.0, 1.9, 2.0, 2.0, 2.0, 1.9, 1.8, 1.6, 1.3, 1.1, 0.8, 0.7, 0.6, 0.7, 0.9, 1.2, 1.5, 1.8, 2.0, 2.1, 2.1, 2.1],
    22: [2.0, 1.9, 1.8, 1.8, 1.9, 2.0, 2.1, 2.1, 2.0, 1.8, 1.5, 1.2, 0.9, 0.7, 0.6, 0.6, 0.8, 1.1, 1.4, 1.8, 2.0, 2.1, 2.1, 2.0],
    23: [1.9, 1.8, 1.7, 1.7, 1.8, 1.9, 2.1, 2.2, 2.2, 2.1, 1.8, 1.4, 1.1, 0.8, 0.6, 0.5, 0.7, 1.0, 1.4, 1.7, 2.0, 2.2, 2.1, 2.0],
    24: [1.8, 1.6, 1.5, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.3, 2.1, 1.7, 1.3, 0.9, 0.7, 0.5, 0.6, 0.9, 1.3, 1.7, 2.0, 2.2, 2.2, 2.0],
    25: [1.8, 1.5, 1.3, 1.2, 1.3, 1.5, 1.8, 2.2, 2.4, 2.4, 2.3, 1.9, 1.5, 1.1, 0.8, 0.6, 0.6, 0.9, 1.2, 1.7, 2.1, 2.3, 2.3, 2.1],
    26: [1.8, 1.5, 1.2, 1.0, 1.0, 1.2, 1.6, 1.9, 2.3, 2.5, 2.4, 2.2, 1.8, 1.3, 0.9, 0.7, 0.7, 0.8, 1.2, 1.6, 2.1, 2.4, 2.5, 2.3],
    27: [2.0, 1.5, 1.2, 0.9, 0.8, 0.9, 1.2, 1.6, 2.0, 2.4, 2.5, 2.3, 2.0, 1.6, 1.1, 0.9, 0.7, 0.9, 1.2, 1.6, 2.0, 2.4, 2.6, 2.5],
    28: [2.2, 1.7, 1.3, 0.9, 0.7, 0.7, 0.9, 1.3, 1.7, 2.1, 2.3, 2.3, 2.1, 1.8, 1.4, 1.0, 0.9, 0.9, 1.1, 1.5, 2.0, 2.4, 2.6, 2.6],
    29: [2.4, 2.0, 1.5, 1.0, 0.7, 0.6, 0.6, 0.9, 1.3, 1.8, 2.1, 2.2, 2.2, 1.9, 1.6, 1.3, 1.1, 1.0, 1.2, 1.5, 1.9, 2.3, 2.6, 2.7],
    30: [2.6, 2.3, 1.8, 1.3, 0.9, 0.6, 0.5, 0.7, 1.0, 1.4, 1.7, 2.0, 2.0, 2.0, 1.7, 1.5, 1.3, 1.2, 1.2, 1.4, 1.8, 2.2, 2.5, 2.7]
}

# 3. CONVERT TABEL MENGJADI DATAFRAME RENTANG WAKTU STREAMLIT
records = []
for day, hours in tabel_big_sep_2026.items():
    for hour_idx, val in enumerate(hours):
        dt = pd.Timestamp(year=2026, month=9, day=day, hour=hour_idx)
        records.append({'datetime': dt, 'water_level': val})

df_big = pd.DataFrame(records).set_index('datetime')

# 4. PREPROCESSING & TRAINING MODEL MACHINE LEARNING
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_big[['water_level']])

LOOKBACK = 12

def create_features(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

X, y = create_features(scaled_data, LOOKBACK)

# Split berdasarkan Jam Sekarang
current_idx = len(df_big[df_big.index <= now_wib]) - LOOKBACK
if current_idx <= 0:
    current_idx = 24 # Fallback

X_train, X_test = X[:current_idx], X[current_idx:]
y_train, y_test = y[:current_idx], y[current_idx:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi ML
predictions = model.predict(X_test)
predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1))

test_timestamps = df_big.index[current_idx + LOOKBACK:]

# 5. VISUALISASI STREAMLIT
fig, ax = plt.subplots(figsize=(12, 5))

# Observasi Aktual Tabel BIG
ax.plot(df_big.index[:current_idx + LOOKBACK], df_big['water_level'][:current_idx + LOOKBACK], 
        label="Aktual Resmi (Tabel BIG 2026)", color="blue", linewidth=1.8)

# Prediksi Machine Learning
ax.plot(test_timestamps, predictions_actual, 
        label="Prediksi ML (Random Forest)", color="red", linestyle="--", linewidth=1.8)

# Garis Penanda Saat Ini
ax.axvline(x=now_wib, color='green', linestyle=':', linewidth=2, 
           label=f'Saat Ini ({now_wib.strftime("%d-%b %H:%M WIB")})')

ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))

ax.set_ylabel("Tinggi Muka Air (Meter)")
ax.set_xlabel("Waktu (WIB)")
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)

plt.xticks(rotation=30)
st.pyplot(fig)

# 6. RINGKASAN METRIK REALTIME
curr_level = df_big.loc[df_big.index <= now_wib, 'water_level'].iloc[-1]
next_level = predictions_actual[0][0]

col1, col2, col3 = st.columns(3)
col1.metric("Level Air Laut Saat Ini (BIG)", f"{curr_level:.2f} m")
col2.metric("Prediksi 1 Jam Ke Depan", f"{next_level:.2f} m")
col3.metric("Kondisi Saat Ini", "Air Laut Surut (Rendah)" if curr_level < 0.5 else "Normal / Pasang")
