import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Pasang Surut BMKG Maritim",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Real-time & Forecast Pasang Surut - BMKG Maritim")
st.markdown("Data diambil dari **maritim.bmkg.go.id** dengan tampilan per jam.")

# Daftar Pelabuhan Populer
PELABUHAN_MAP = {
    "Pelabuhan Probolinggo": "pelabuhan-probolinggo",
    "Pelabuhan Tanjung Perak (Surabaya)": "pelabuhan-tanjung-perak",
    "Pelabuhan Tanjung Emas (Semarang)": "pelabuhan-tanjung-emas",
    "Pelabuhan Tanjung Priok (Jakarta)": "pelabuhan-tanjung-priok",
    "Pelabuhan Belawan (Medan)": "pelabuhan-belawan"
}

# Sidebar - Pilih Pelabuhan
st.sidebar.header("📍 Lokasi Stasiun/Pelabuhan")
selected_port_label = st.sidebar.selectbox("Pilih Pelabuhan:", list(PELABUHAN_MAP.keys()))
port_slug = PELABUHAN_MAP[selected_port_label]

# Fungsi Scrape Data BMKG Maritim
@st.cache_data(ttl=1800)
def fetch_bmkg_tide_data(slug):
    url = f"https://maritim.bmkg.go.id/cuaca/pelabuhan/{slug}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Gagal mengambil data dari BMKG (Status HTTP: {response.status_code})"
        
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        
        rows = []
        for line in text.split("\n"):
            line_str = line.strip()
            if " Pasang Surut (m)" in line_str or "%" in line_str:
                parts = line_str.split()
                for p in parts:
                    if p.endswith("m") and ("+" in p or "-" in p):
                        try:
                            val = float(p.replace("m", "").replace("+", ""))
                            rows.append(val)
                        except ValueError:
                            pass
        
        if not rows:
            return None, "Format data pasang surut tidak ditemukan pada BMKG."
            
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        times = [start_time + pd.Timedelta(hours=i) for i in range(len(rows))]
        
        df = pd.DataFrame({
            "Waktu": times,
            "Elevasi (m)": rows
        })
        return df, None

    except Exception as e:
        return None, f"Terjadi kesalahan saat memuat data: {str(e)}"

# Ambil Data
with st.spinner(f"Memuat data per jam untuk {selected_port_label}..."):
    df_tide, error_msg = fetch_bmkg_tide_data(port_slug)

if error_msg:
    st.info("💡 **Mode Simulasi Per Jam:** Menampilkan simulasi data jam demi jam.")
    t = np.arange(48)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now + pd.Timedelta(hours=i) for i in range(48)]
    elevation = 1.2 * np.cos(2 * np.pi * t / 12.42) + 0.4 * np.cos(2 * np.pi * t / 23.93)
    df_tide = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevation})

# Indikator Utama
col1, col2, col3 = st.columns(3)
col1.metric("Pasang Tertinggi (HWL)", f"{df_tide['Elevasi (m)'].max():.2f} m")
col2.metric("Muka Air Rata-rata (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
col3.metric("Surut Terendah (LWL)", f"{df_tide['Elevasi (m)'].min():.2f} m")

st.divider()

# Visualisasi Grafik Detail Per Jam
st.subheader(f"📈 Grafik Pasang Surut (Interval Per Jam) - {selected_port_label}")

fig, ax = plt.subplots(figsize=(14, 5))

# Plot Garis Elevasi
ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2, marker="o", markersize=3, label="Elevasi Air Laut (m)")
ax.axhline(0, color="red", linestyle="--", alpha=0.6, label="MSL (0.0 m)")

# Format Sumbu X untuk Tampilan Jam (Contoh: 09 Sep 14:00)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Tampilkan label tiap 3 jam
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Garis grid kecil per 1 jam
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

ax.set_ylabel("Elevasi (Meter)", fontsize=11)
ax.set_xlabel("Waktu (Tanggal & Jam)", fontsize=11)
ax.grid(True, which="major", linestyle="--", alpha=0.7)
ax.grid(True, which="minor", linestyle=":", alpha=0.4)
ax.legend(loc="upper right")
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Format Tabel untuk Menampilkan Tanggal & Jam dengan Jelas
df_tabel = df_tide.copy()
df_tabel["Waktu"] = df_tabel["Waktu"].dt.strftime("%d-%m-%Y %H:%M WIB")

with st.expander("📄 lihat Tabel Data Pasang Surut Per Jam"):
    st.dataframe(df_tabel, use_container_width=True)
