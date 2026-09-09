import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Prediksi Water Intake PLTGU / Pasang Surut",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Aplikasi Prediksi Pasang Surut & Water Intake PLTGU")
st.write("Lokasi Acuan: **Stasiun Probolinggo**")

# 1. PARAMETER HARMONIK (Data Probolinggo)
S0 = 4.10  # Mean Sea Level (m)

harmonics = {
    'M2':  {'A': 0.20, 'g': 247.0, 'speed': 28.9841042},
    'S2':  {'A': 0.16, 'g': 288.0, 'speed': 30.0000000},
    'N2':  {'A': 0.05, 'g': 231.0, 'speed': 28.4397295},
    'K2':  {'A': 0.04, 'g': 288.0, 'speed': 30.0821373},
    'K1':  {'A': 0.53, 'g': 250.0, 'speed': 15.0410686},
    'O1':  {'A': 0.38, 'g': 187.0, 'speed': 13.9430356},
    'P1':  {'A': 0.18, 'g': 250.0, 'speed': 14.9589314},
    'M4':  {'A': 0.01, 'g': 180.0, 'speed': 57.9682084},
    'MS4': {'A': 0.01, 'g': 220.0, 'speed': 58.9841042}
}

# Sidebar Input User
st.sidebar.header("⚙️ Pengaturan Prediksi")
tanggal_mulai = st.sidebar.date_input("Tanggal Mulai", datetime.now())
jam_mulai = st.sidebar.time_input("Jam Mulai", datetime.now().time())
durasi_hari = st.sidebar.slider("Durasi Prediksi (Hari)", min_value=1, max_value=7, value=2)

# Menggabungkan Date & Time
start_datetime = datetime.combine(tanggal_mulai, jam_mulai)
total_jam = durasi_hari * 24

# 2. PROSES PERHITUNGAN
waktu_list = []
tinggi_list = []

for h in range(total_jam):
    current_time = start_datetime + timedelta(hours=h)
    
    # Formula Pasut
    height = S0
    for constituent, data in harmonics.items():
        A = data['A']
        g = np.radians(data['g'])
        speed = np.radians(data['speed'])
        height += A * np.cos(speed * h - g)
        
    waktu_list.append(current_time)
    tinggi_list.append(height)

# DataFrame Hasil
df = pd.DataFrame({
    'Waktu': waktu_list,
    'Tinggi Air (m)': tinggi_list
})

# 3. TAMPILKAN METRIK KUNCI
col1, col2, col3 = st.columns(3)
col1.metric("Pasang Maksimum", f"{df['Tinggi Air (m)'].max():.2f} m")
col2.metric("Rata-Rata (MSL)", f"{S0:.2f} m")
col3.metric("Surut Terendah", f"{df['Tinggi Air (m)'].min():.2f} m")

st.divider()

# 4. TAMPILKAN GRAFIK
st.subheader("📊 Grafik Prediksi Elevasi Air")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Waktu'], df['Tinggi Air (m)'], color='#1f77b4', linewidth=2, label='Tinggi Air')
ax.axhline(y=S0, color='red', linestyle='--', label=f'MSL ({S0} m)')
ax.set_ylabel("Tinggi (m)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig)

# 5. TABEL DATA & DOWNLOAD
st.subheader("📋 Tabel Data Prediksi")
st.dataframe(df, use_container_width=True)

# Button Download CSV
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Data CSV",
    data=csv,
    file_name=f"prediksi_pasut_{tanggal_mulai}.csv",
    mime="text/csv"
)
