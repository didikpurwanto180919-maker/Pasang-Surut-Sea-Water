import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
st.markdown("Data diambil langsung dari situs resmi **maritim.bmkg.go.id**.")

# Daftar Pelabuhan Populer di BMKG Maritim
PELABUHAN_MAP = {
    "Pelabuhan Probolinggo": "pelabuhan-probolinggo",
    "Pelabuhan Tanjung Perak (Surabaya)": "pelabuhan-tanjung-perak",
    "Pelabuhan Tanjung Emas (Semarang)": "pelabuhan-tanjung-emas",
    "Pelabuhan Tanjung Priok (Jakarta)": "pelabuhan-tanjung-priok",
    "Pelabuhan Tanjung Api-Api": "pelabuhan-tanjung-api-api",
    "Pelabuhan Belawan (Medan)": "pelabuhan-belawan"
}

# Sidebar - Pilih Pelabuhan
st.sidebar.header("📍 Lokasi Stasiun/Pelabuhan")
selected_port_label = st.sidebar.selectbox("Pilih Pelabuhan:", list(PELABUHAN_MAP.keys()))
port_slug = PELABUHAN_MAP[selected_port_label]

# Fungsi Scrape Data BMKG Maritim
@st.cache_data(ttl=1800)  # Simpan cache selama 30 menit
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
        
        # Ekstrak data teks dari halaman
        text = soup.get_text()
        
        # Menguraikan baris yang memuat info Pasang Surut
        # Format string BMKG biasanya memuat baris dengan pola: 'X Sep 26, HH.00' diikuti 'Pasang Surut (m)'
        rows = []
        for line in text.split("\n"):
            line_str = line.strip()
            if " Pasang Surut (m)" in line_str or "%" in line_str:
                parts = line_str.split()
                # Mencari pola nilai elevasi (misal +0.87 m atau -1.05 m)
                for i, p in enumerate(parts):
                    if p.endswith("m") and ("+" in p or "-" in p):
                        try:
                            val = float(p.replace("m", "").replace("+", ""))
                            rows.append(val)
                        except ValueError:
                            pass
        
        if not rows:
            return None, "Format data pasang surut tidak ditemukan atau sedang diubah oleh BMKG."
            
        # Buat time series sederhana berdasarkan durasi data jam demi jam dari BMKG
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
with st.spinner(f"Memuat data real-time BMKG untuk {selected_port_label}..."):
    df_tide, error_msg = fetch_bmkg_tide_data(port_slug)

if error_msg:
    st.error(error_msg)
    st.info("💡 **Mode Simulasi Aktif:** Menampilkan estimasi matematis karena data BMKG tidak dapat dijangkau saat ini.")
    
    # Fallback / Simulasi jika scraping terhalang
    t = np.arange(48)
    now = datetime.now()
    times = [now + pd.Timedelta(hours=i) for i in range(48)]
    elevation = 1.0 * np.cos(2 * np.pi * t / 12.42) + 0.5 * np.cos(2 * np.pi * t / 23.93)
    df_tide = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevation})

# Tampilkan Indikator
col1, col2, col3 = st.columns(3)
col1.metric("Pasang Tertinggi (HWL)", f"{df_tide['Elevasi (m)'].max():.2f} m")
col2.metric("Muka Air Rata-rata (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
col3.metric("Surut Terendah (LWL)", f"{df_tide['Elevasi (m)'].min():.2f} m")

st.divider()

# Visualisasi Grafik
st.subheader(f"📈 Grafik Pasang Surut - {selected_port_label}")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2.5, label="Elevasi Air Laut (m)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.7, label="MSL (0.0 m)")

ax.set_ylabel("Ketinggian (Meter)")
ax.set_xlabel("Waktu")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")
plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig)

# Tabel Data
with st.expander("📄 Lihat Tabel Detail per Jam"):
    st.dataframe(df_tide, use_container_width=True)
