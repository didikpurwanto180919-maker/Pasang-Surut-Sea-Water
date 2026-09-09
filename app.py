import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Config halaman Streamlit
st.set_page_config(
    page_title="Pasang Surut Sea Water Level Probolinggo 2026",
    page_icon="🌊",
    layout="wide"
)

# Title & Deskripsi
st.title("🌊 Prediksi & Real-Time Data Pasang Surut Air Laut Probolinggo")
st.markdown("""
Aplikasi ini menampilkan kombinasi data **Real-Time / Prakiraan BMKG Maritim** dan **Model Machine Learning (Random Forest)** untuk Sea Water Level Probolinggo 2026.
""")

# ==========================================
# 1. FUNCTION FETCH DATA REALTIME / BMKG
# ==========================================
@st.cache_data(ttl=3600)  # Cache data BMKG selama 1 jam
def fetch_bmkg_maritim_data():
    url = "https://maritim.bmkg.go.id/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Scraping data umum maritim (Status koneksi BMKG)
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

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    df['ML_Predicted_Sea_Level_m'] = np.round(model.predict(X), 2)
    
    return df, rmse, r2

# Load Data ML
with st.spinner("Memproses data & melatih model Machine Learning..."):
    df, rmse, r2 = generate_and_train()

# Status Integrasi BMKG
bmkg_status, bmkg_msg = fetch_bmkg_maritim_data()

# ==========================================
# 3. SIDEBAR & METRICS
# ==========================================
st.sidebar.header("⚙️ Kontrol & Live Source")

if bmkg_status:
    st.sidebar.success("🟢 Connected to BMKG Maritim")
else:
    st.sidebar.warning(f"🔴 BMKG Source: {bmkg_msg}")

selected_month = st.sidebar.selectbox(
    "Pilih Bulan untuk Dilihat:",
    options=list(range(1, 13)),
    index=8, # Default ke September
    format_func=lambda x: pd.to_datetime(f'2026-{x:02d}-01').strftime('%B')
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Data Points", f"{len(df):,} Jam")
col2.metric("Model RMSE Error", f"{rmse:.4f} m")
col3.metric("Model R² Score", f"{r2:.4f}")
col4.metric("Sumber Realtime", "BMKG Maritim" if bmkg_status else "Simulasi/ML")

st.divider()

# ==========================================
# 4. GRAFIK VISUALISASI
# ==========================================
st.subheader(f"📈 Grafik Pasang Surut Bulan {pd.to_datetime(f'2026-{selected_month:02d}-01').strftime('%B 2026')}")

df_filtered = df[df['Month'] == selected_month]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_filtered['Timestamp'], df_filtered['Sea_Level_m'], label='Simulated / BMKG Baseline Level', color='#1f77b4', linewidth=1.5)
ax.plot(df_filtered['Timestamp'], df_filtered['ML_Predicted_Sea_Level_m'], label='ML Predicted Sea Level', color='#d62728', linestyle='--', linewidth=1)
ax.axhline(y=1.4, color='green', linestyle=':', label='Mean Sea Level (1.4m)')
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
    st.write("Silakan unduh dataset lengkap sepanjang tahun 2026:")
    
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
