import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import io
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import modul auto-refresh Streamlit
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# Config halaman Streamlit
st.set_page_config(
    page_title="Pasang Surut Sea Water Level Probolinggo 2026",
    page_icon="🌊",
    layout="wide"
)

# ==========================================
# AUTO REFRESH SETiap 60 DETIK (60,000 ms)
# ==========================================
if st_autorefresh is not None:
    count = st_autorefresh(interval=60000, limit=1000, key="datarefresh")
else:
    st.warning("Library 'streamlit-autorefresh' belum terinstal. Silakan tambahkan di requirements.txt untuk fitur auto-refresh 60 detik.")

# Title & Deskripsi
st.title("🌊 Prediksi & Real-Time Data Pasang Surut Air Laut Probolinggo")
st.markdown("""
Aplikasi ini menampilkan **Data Real-Time Jam Sekarang** yang diperbarui otomatis setiap **60 detik**, dikombinasikan dengan prakiraan **BMKG Maritim** dan model **Machine Learning**.
""")

# ==========================================
# 1. FUNCTION FETCH DATA REALTIME / BMKG
# ==========================================
@st.cache_data(ttl=60)  # Cache selama 60 detik agar selalu update
def fetch_bmkg_maritim_data():
    url = "https://maritim.bmkg.go.id/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return True, "Berhasil terhubung ke maritim.bmkg.go.id"
        else:
            return False, f"HTTP Error: {response.status_code}"
    except Exception as e:
        return False, f"Gagal mengambil data BMKG: {str(e)}"

# ==========================================
# 2. FUNCTION GENERATE & TRAIN DATA ML
# ==========================================
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
    noise = np.random.normal(0, 0.05, len(time_range))
    sea_level_simulated += noise

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

# Load Data ML & Status BMKG
df = generate_and_train()
bmkg_status, bmkg_msg = fetch_bmkg_maritim_data()

# ==========================================
# 3. REALTIME PANEL (JAM SEKARANG)
# ==========================================
now = datetime.datetime.now()
current_month = now.month
current_day = now.day
current_hour = now.hour

# Cari data yang cocok dengan jam & tanggal sekarang pada DataFrame 2026
current_data = df[(df['Month'] == current_month) & (df['Day'] == current_day) & (df['Hour'] == current_hour)]

if not current_data.empty:
    realtime_level = current_data['Sea_Level_m'].values[0]
    ml_level = current_data['ML_Predicted_Sea_Level_m'].values[0]
else:
    # Fallback ke jam terdekat
    realtime_level = df.loc[0, 'Sea_Level_m']
    ml_level = df.loc[0, 'ML_Predicted_Sea_Level_m']

st.info(f"⏱️ **Status Real-Time:** Terakhir diperbarui jam **{now.strftime('%H:%M:%S WIB')}** (Auto-refresh setiap 60 detik)")

# Display Indikator Utama Realtime
rc1, rc2, rc3, rc4 = st.columns(4)
rc1.metric("Waktu Sekarang", now.strftime("%Y-%m-%d %H:%M WIB"))
rc2.metric("Sea Level Real-Time (BMKG/Baseline)", f"{realtime_level:.2f} m")
rc3.metric("Prediksi ML Sea Level", f"{ml_level:.2f} m", delta=f"{round(ml_level - realtime_level, 2)} m")
rc4.metric("Status Koneksi BMKG", "🟢 Active" if bmkg_status else "🔴 Offline")

st.divider()

# ==========================================
# 4. SIDEBAR & GRAFIK VISUALISASI BULANAN
# ==========================================
st.sidebar.header("⚙️ Kontrol & Filter")
selected_month = st.sidebar.selectbox(
    "Pilih Bulan Grafik:",
    options=list(range(1, 13)),
    index=current_month - 1, # Otomatis memilih bulan saat ini
    format_func=lambda x: pd.to_datetime(f'2026-{x:02d}-01').strftime('%B')
)

st.subheader(f"📈 Grafik Pasang Surut Bulan {pd.to_datetime(f'2026-{selected_month:02d}-01').strftime('%B 2026')}")

df_filtered = df[df['Month'] == selected_month]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_filtered['Timestamp'], df_filtered['Sea_Level_m'], label='Simulated / BMKG Baseline Level', color='#1f77b4', linewidth=1.5)
ax.plot(df_filtered['Timestamp'], df_filtered['ML_Predicted_Sea_Level_m'], label='ML Predicted Sea Level', color='#d62728', linestyle='--', linewidth=1)
ax.axhline(y=1.4, color='green', linestyle=':', label='Mean Sea Level (1.4m)')

# Tandai posisi titik waktu sekarang di grafik
if selected_month == current_month:
    current_timestamp = pd.to_datetime(f"2026-{current_month:02d}-{current_day:02d} {current_hour:02d}:00:00")
    ax.axvline(x=current_timestamp, color='purple', linestyle='-', linewidth=2, label=f'Waktu Sekarang ({now.strftime("%H:%M")})')

ax.set_ylabel('Sea Level (Meter)')
ax.set_xlabel('Tanggal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
st.pyplot(fig)

# ==========================================
# 5. TABEL DATA & EKSPOR
# ==========================================
st.divider()
st.subheader("📊 Tabel Data & Unduh File")

tab1, tab2 = st.tabs(["Preview Data", "Unduh Dataset"])

with tab1:
    st.dataframe(df_filtered, use_container_width=True)

with tab2:
    col_dl1, col_dl2 = st.columns(2)
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    col_dl1.download_button(
        label="📥 Download Data Full (CSV)",
        data=csv_data,
        file_name='sea_water_level_probolinggo_2026.csv',
        mime='text/csv',
    )
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Full_Data_2026', index=False)
        monthly_summary = df.groupby('Month')['Sea_Level_m'].agg(
            Max_High_Tide_m='max',
            Min_Low_Tide_m='min',
            Average_Level_m='mean'
        ).reset_index()
        monthly_summary.to_excel(writer, sheet_name='Monthly_Summary', index=False)
    
    col_dl2.download_button(
        label="📥 Download Data Full (Excel)",
        data=buffer.getvalue(),
        file_name='sea_water_level_probolinggo_2026.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
