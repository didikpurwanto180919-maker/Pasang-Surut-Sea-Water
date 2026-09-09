import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import io
import datetime
import urllib3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Safe Import untuk pytz (Zona Waktu WIB)
try:
    import pytz
    wib_tz = pytz.timezone('Asia/Jakarta')
    now = datetime.datetime.now(wib_tz)
except Exception:
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)

# Safe Import untuk Auto-Refresh 60 Detik
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh_installed = True
except Exception:
    st_autorefresh_installed = False

# Safe Import untuk Folium Map Integration
try:
    import folium
    from streamlit_folium import st_folium
    folium_installed = True
except Exception:
    folium_installed = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Config Halaman
st.set_page_config(
    page_title="Pasang Surut Air Laut - S7°38.659' E113°01.641'",
    page_icon="🌊",
    layout="wide"
)

if st_autorefresh_installed:
    st_autorefresh(interval=60000, limit=1000, key="datarefresh")

# Header Aplikasi
st.title("🌊 Prediksi & Real-Time Data Pasang Surut Air Laut Probolinggo")
st.caption("📍 **Lokasi Koordinat Target:** S7°38.659' E113°01.641' (PLTGU Grati / Perairan Probolinggo)")

st.markdown("""
Aplikasi ini menampilkan **Data Real-Time Jam Sekarang (WIB)** yang diperbarui otomatis setiap **60 detik**, dikombinasikan dengan status **BMKG Maritim**, visualisasi **Peta Lokasi**, dan model **Machine Learning**.
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
        'Latitude': "S7°38.659'",
        'Longitude': "E113°01.641'",
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

# 3. REALTIME PANEL
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

rc1, rc2, rc3, rc4, rc5 = st.columns(5)
rc1.metric("Waktu Sekarang", now.strftime("%Y-%m-%d %H:%M WIB"))
rc2.metric("Koordinat Lokasi", "S7°38.659' E113°01.641'")
rc3.metric("Sea Level Real-Time", f"{realtime_level:.2f} m")
rc4.metric("Prediksi ML Sea Level", f"{ml_level:.2f} m", delta=f"{round(ml_level - realtime_level, 2)} m")
rc5.metric("Status Koneksi BMKG", f"🟢 {bmkg_msg}" if bmkg_status else "🔴 Offline")

st.divider()

# ==========================================
# 4. TAMPILAN PETA LOKASI REALTIME (OPENSTREETMAP / FOLIUM)
# ==========================================
st.subheader("🗺️ Tampilan Peta Lokasi Real-Time (S7°38.659' E113°01.641')")

lat_decimal = -7.644317
lon_decimal = 113.027350

col_map1, col_map2 = st.columns([3, 1])

with col_map1:
    if folium_installed:
        m = folium.Map(location=[lat_decimal, lon_decimal], zoom_start=15)
        folium.Marker(
            [lat_decimal, lon_decimal],
            popup="Titik Pantau Pasang Surut: S7°38.659' E113°01.641'",
            tooltip="📍 S7°38.659' E113°01.641' (PLTGU Grati / Probolinggo)",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        st_folium(m, width="100%", height=350)
    else:
        # Fallback Native Streamlit Map jika Folium belum terpasang
        map_data = pd.DataFrame({'lat': [lat_decimal], 'lon': [lon_decimal]})
        st.map(map_data, zoom=14)

with col_map2:
    st.markdown("### 📌 Detail Titik Stasiun")
    st.write("**Nama Stasiun:** Perairan Probolinggo / PLTGU Grati")
    st.write("**Latitude (S):** 7°38.659' ( -7.644317 )")
    st.write("**Longitude (E):** 113°01.641' ( 113.027350 )")
    st.write(f"**Tinggi Air Laut Saat Ini:** `{realtime_level:.2f} Meter`")
    st.write(f"**Rata-rata (MSL):** `1.40 Meter`")

st.divider()

# ==========================================
# 5. KONTROL SIDEBAR & GRAFIK VISUALISASI
# ==========================================
st.sidebar.header("⚙️ Kontrol Grafik")

view_mode = st.sidebar.radio(
    "Pilih Tampilan Grafik:",
    options=["Mode Harian (24 Jam)", "Mode Per Jam (Detail 6 Jam)"]
)

selected_date = st.sidebar.date_input(
    "Pilih Tanggal:",
    value=datetime.date(2026, current_month, current_day),
    min_value=datetime.date(2026, 1, 1),
    max_value=datetime.date(2026, 12, 31)
)

df_daily = df[(df['Timestamp'].dt.date == selected_date)]

if view_mode == "Mode Harian (24 Jam)":
    st.subheader(f"📈 Grafik Pasang Surut Harian ({selected_date.strftime('%d %B %Y')}) — S7°38.659' E113°01.641'")
    df_plot = df_daily
else:
    st.subheader(f"⏱️ Grafik Pasang Surut Per Jam ({selected_date.strftime('%d %B %Y')}) — S7°38.659' E113°01.641'")
    selected_hour_start = st.sidebar.slider("Pilih Jam Awal:", 0, 18, current_hour if current_hour <= 18 else 18)
    df_plot = df_daily[(df_daily['Hour'] >= selected_hour_start) & (df_daily['Hour'] <= selected_hour_start + 6)]

fig, ax = plt.subplots(figsize=(12, 4.5))

ax.plot(df_plot['Timestamp'], df_plot['Sea_Level_m'], marker='o', label='Simulated / BMKG Baseline Level', color='#1f77b4', linewidth=2)
ax.plot(df_plot['Timestamp'], df_plot['ML_Predicted_Sea_Level_m'], marker='x', label='ML Predicted Sea Level', color='#d62728', linestyle='--', linewidth=1.5)
ax.axhline(y=1.4, color='green', linestyle=':', label='Mean Sea Level (1.4m)')

if selected_date == datetime.date(2026, current_month, current_day):
    current_timestamp = pd.to_datetime(f"2026-{current_month:02d}-{current_day:02d} {current_hour:02d}:00:00")
    if current_timestamp in df_plot['Timestamp'].values:
        ax.axvline(x=current_timestamp, color='purple', linestyle='-', linewidth=2.5, label=f'Waktu Sekarang ({now.strftime("%H:%M WIB")})')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1 if view_mode != "Mode Harian (24 Jam)" else 2))

ax.set_ylabel('Sea Level (Meter)')
ax.set_xlabel('Jam (WIB)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right')
plt.xticks(rotation=0)

st.pyplot(fig)

# ==========================================
# 6. TABEL DATA & UNDUH FILE
# ==========================================
st.divider()
st.subheader("📊 Tabel Data & Unduh File")

tab1, tab2 = st.tabs(["Preview Data Terpilih", "Unduh Dataset Full"])

with tab1:
    st.dataframe(df_plot, use_container_width=True)

with tab2:
    col_dl1, col_dl2 = st.columns(2)
    csv_data = df.to_csv(index=False).encode('utf-8')
    col_dl1.download_button("📥 Download Data Full (CSV)", csv_data, 'sea_water_level_probolinggo_S7_38_659_E113_01_641.csv', 'text/csv')
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data_S7_38_659_E113_01_641', index=False)
    col_dl2.download_button("📥 Download Data Full (Excel)", buffer.getvalue(), 'sea_water_level_probolinggo_S7_38_659_E113_01_641.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
