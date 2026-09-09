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

st.title("🌊 Dashboard Harian Level Air Laut")
st.subheader("Lokasi: Probolinggo / Pasuruan (S 07° 44' E 113° 12') - GMT+07.00")

# 1. WAKTU REALTIME
wib_tz = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(wib_tz).replace(tzinfo=None)

# 2. SIDEBAR: FITUR PILIH TANGGAL HARIAN
st.sidebar.header("🗓️ Kontrol Dashboard")
selected_date = st.sidebar.date_input("Pilih Tanggal Visualisasi:", value=now_wib.date())

st.sidebar.markdown(f"**Waktu Realtime:**\n`{now_wib.strftime('%Y-%m-%d %H:%M:%S WIB')}`")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# 3. TABEL AKTUAL RESMI BIG (SEPTEMBER 2026)
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

# 4. FILTER DATA KHUSUS HARIAN
records = []
for day, hours in tabel_big_sep_2026.items():
    for hour_idx, val in enumerate(hours):
        dt = pd.Timestamp(year=2026, month=9, day=day, hour=hour_idx)
        records.append({'datetime': dt, 'water_level': val})

df_all = pd.DataFrame(records).set_index('datetime')

# Filter berdasarkan tanggal yang dipilih di sidebar
start_day = pd.Timestamp(selected_date)
end_day = start_day + pd.Timedelta(hours=23, minutes=59)

df_daily = df_all[(df_all.index >= start_day) & (df_all.index <= end_day)]

# 5. MODEL MACHINE LEARNING UNTUK SKALA HARIAN
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_all[['water_level']])

LOOKBACK = 6

def create_features(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

X, y = create_features(scaled_data, LOOKBACK)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediksi untuk hari yang dipilih
scaled_daily = scaler.transform(df_daily[['water_level']])
X_daily = []
for i in range(len(scaled_daily)):
    idx = df_all.index.get_loc(df_daily.index[i])
    if idx >= LOOKBACK:
        X_daily.append(scaled_data[idx-LOOKBACK:idx, 0])
    else:
        X_daily.append(scaled_data[:LOOKBACK, 0])

X_daily = np.array(X_daily)
pred_daily = model.predict(X_daily)
pred_daily_actual = scaler.inverse_transform(pred_daily.reshape(-1, 1))

# 6. VISUALISASI DASBOR HARIAN (Sumbu X Sangat Rapi & Jelas)
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df_daily.index, df_daily['water_level'], label="Aktual Resmi (Tabel BIG)", color="blue", linewidth=2, marker='o', markersize=4)
ax.plot(df_daily.index, pred_daily_actual, label="Prediksi ML (Random Forest)", color="red", linestyle="--", linewidth=1.8)

# Garis Penanda Jika Hari Ini yang Dipilih
if selected_date == now_wib.date():
    ax.axvline(x=now_wib, color='green', linestyle=':', linewidth=2, label=f'Saat Ini ({now_wib.strftime("%H:%M WIB")})')

# Format Sumbu-X Per Jam
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1)) # Tampilkan setiap jam (00:00 - 23:00)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax.set_ylabel("Tinggi Muka Air (Meter)")
ax.set_xlabel(f"Jam (WIB) - Tanggal {selected_date.strftime('%d %B %Y')}")
ax.legend(loc="upper right")
ax.grid(True, linestyle=":", alpha=0.6)

plt.xticks(rotation=45)
st.pyplot(fig)

# 7. METRIK RINGKASAN HARIAN
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pasang Maksimum Hari Ini", f"{df_daily['water_level'].max():.2f} m")
col2.metric("Surut Minimum Hari Ini", f"{df_daily['water_level'].min():.2f} m")
col3.metric("Rata-rata Muka Air (MSL)", f"{df_daily['water_level'].mean():.2f} m")

if selected_date == now_wib.date():
    curr_val = df_all.loc[df_all.index <= now_wib, 'water_level'].iloc[-1]
    col4.metric("Level Jam Ini", f"{curr_val:.2f} m")
else:
    col4.metric("Total Data", f"{len(df_daily)} Jam")
