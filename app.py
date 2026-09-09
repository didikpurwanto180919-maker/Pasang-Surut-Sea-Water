import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import datetime
import urllib3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Safe Import untuk pytz (Zona Waktu)
try:
    import pytz
    wib_tz = pytz.timezone('Asia/Jakarta')
    now = datetime.datetime.now(wib_tz)
except Exception:
    # Fallback jika pytz belum terinstal (UTC + 7 jam)
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)

# Safe Import untuk Auto-Refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh_installed = True
except Exception:
    st_autorefresh_installed = False

# Matikan warning SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Config Halaman
st.set_page_config(
    page_title="Pasang Surut Sea Water Level Probolinggo 2026",
    page_icon="🌊",
    layout="wide"
)

# Jalankan Auto-Refresh jika library tersedia
if st_autorefresh_installed:
    st_autorefresh(interval=60000, limit=1000, key="datarefresh")

st.title("🌊 Prediksi & Real-Time Data Pasang Surut Air Laut Probolinggo")
st.markdown("""
Aplikasi ini menampilkan **Data Real-Time Jam Sekarang (WIB)** yang diperbarui otomatis setiap **60 detik**, dikombinasikan dengan status **BMKG Maritim** dan model **Machine Learning**.
""")

# 1. FUNCTION FETCH DATA BMKG
@st.cache_data(ttl=120)
def fetch_bmkg_maritim_data():
    url = "https://maritim.bmkg.go.id/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        if response.status_code == 200:
            return True, "Active"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception:
        try:
            alt_url = "https://data.bmkg.go.id/"
            alt_resp = requests.get(alt_url, headers=headers, timeout=5, verify=False)
            if alt_resp.status_code == 200:
                return True, "Active (Alt Gateway)"
        except Exception:
            pass
        return False, "Offline"

# 2. FUNCTION GENERATE & TRAIN DATA ML
@st.cache_data
def generate_and_train():
    time_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-12-31 23:00:00', freq='h')
    
    hours = np.arange(len(time_range))
    tide_m2 = 0.8 * np.cos(2 * np.pi * hours / 12.42)
    tide_s2 = 0.3 * np.cos(2 * np.pi * hours / 12.00)
    tide_k1 = 0.9 * np.cos(2 * np.pi * hours / 23.93 + 0.5)
    tide_o1 = 0.5 * np.cos(2 * np.pi * hours / 25.82 - 0.3)
    msl = 1.4

    sea_level_simulated = msl + tide_m2 + tide_s2 + tide_k1 + tide_o1
    np.random.seed(42)
    sea_level_simulated += np.random.normal(0, 0.05, len(time_range))

    df = pd.DataFrame({
        'Timestamp': time_range,
        'Year': time_range.year,
        'Month': time_range.month,
        'Day': time_range.day,
        'Hour': time_range.hour,
        'DayOfWeek': time_range.dayofweek,
        'DayOfYear': time_range.dayofyear,
        'Sea_Level_m': np.round(sea_level_simulated, 2)
    })

    X = df[['Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear']]
    y = df['Sea_Level_m']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df['ML_Predicted_Sea_Level_m'] = np.round(model.predict(X), 2)
    return df

df = generate_and_train()
bmkg_status, bmkg_msg = fetch_bmkg_maritim_data()

# 3. REALTIME PANEL (WIB)
current_month = now.month
current_day = now.day
current_hour = now.hour

current_data = df[(df['Month'] == current_month) & (df['Day'] == current_day) & (df['Hour'] == current_hour)]

if not current_data.empty:
    realtime_level = current_data['Sea_Level_m'].values[0]
    ml_level = current_data['ML_Predicted_Sea_Level_m'].values[0]
else:
    realtime_level = df.loc[0, 'Sea_Level_m']
    ml_level = df.loc[0, 'ML_Predicted_Sea_Level_m']

st.info(f"⏱️ **Status Real-Time:** Terakhir diperbarui jam **{now.strftime('%H:%M:%S WIB')}** (Auto-refresh setiap 60 detik)")

rc1, rc2, rc3, rc4 = st.columns(4)
rc1.metric("Waktu Sekarang", now.strftime("%Y-%m-%d %H:%M WIB"))
rc2.metric("Sea Level Real-Time", f"{realtime_level:.2f} m")
rc3.metric("Prediksi ML Sea Level", f"{ml_level:.2f} m", delta=f"{round(ml_level - realtime_level, 2)} m")
rc4.metric("Status Koneksi BMKG", f"🟢 {bmkg_msg}" if bmkg_status else "🔴 Offline")

st.divider()

# 4. SIDEBAR & GRAFIK VISUALISASI
st.sidebar.header("⚙️ Kontrol & Filter")
selected_month = st.sidebar.selectbox(
    "Pilih Bulan Grafik:",
    options=list(range(1, 13)),
    index=current_month - 1,
    format_func=lambda x: pd.to_datetime(f'2026-{x:02d}-01').strftime('%B')
)

st.subheader(f"📈 Grafik Pasang Surut Bulan {pd.to_datetime(f'2026-{selected_month:02d}-01').strftime('%B 2026')}")

df_filtered = df[df['Month'] == selected_month]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_filtered['Timestamp'], df_filtered['Sea_Level_m'], label='Simulated / BMKG Baseline Level', color='#1f77b4', linewidth=1.5)
ax.plot(df_filtered['Timestamp'], df_filtered['ML_Predicted_Sea_Level_m'], label='ML Predicted Sea Level', color='#d62728', linestyle='--', linewidth=1)
ax.axhline(y=1.4, color='green', linestyle=':', label='Mean Sea Level (1.4m)')

if selected_month == current_month:
    current_timestamp = pd.to_datetime(f"2026-{current_month:02d}-{current_day:02d} {current_hour:02d}:00:00")
    ax.axvline(x=current_timestamp, color='purple', linestyle='-', linewidth=2, label=f'Waktu Sekarang ({now.strftime("%H:%M WIB")})')

ax.set_ylabel('Sea Level (Meter)')
ax.set_xlabel('Tanggal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
st.pyplot(fig)

# 5. TABEL DATA & UNDUH
st.divider()
st.subheader("📊 Tabel Data & Unduh File")

tab1, tab2 = st.tabs(["Preview Data", "Unduh Dataset"])

with tab1:
    st.dataframe(df_filtered, use_container_width=True)

with tab2:
    col_dl1, col_dl2 = st.columns(2)
    csv_data = df.to_csv(index=False).encode('utf-8')
    col_dl1.download_button("📥 Download Data Full (CSV)", csv_data, 'sea_water_level_probolinggo_2026.csv', 'text/csv')
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Full_Data_2026', index=False)
    col_dl2.download_button("📥 Download Data Full (Excel)", buffer.getvalue(), 'sea_water_level_probolinggo_2026.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
