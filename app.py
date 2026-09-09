import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

# Konfigurasi Halaman Utama Streamlit
st.set_page_config(
    page_title="BMKG Realtime Pasut - Perairan Pasuruan",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Realtime Data Pasang Surut - BMKG Maritim")
st.caption("Lokasi Target: **Perairan Pasuruan / Pelabuhan Probolinggo - Selat Madura (-7.63978, 113.03175)**")

# -------------------------------------------------------------------
# SCRAPER BMKG MARITIM REALTIME DATA
# -------------------------------------------------------------------
@st.cache_data(ttl=900)  # Refresh cache otomatis setiap 15 menit
def get_bmkg_maritim_data():
    # URL BMKG Maritim khusus wilayah perairan Pasuruan & Probolinggo
    urls = [
        "https://maritim.bmkg.go.id/cuaca/perairan/perairan-pasuruan",
        "https://maritim.bmkg.go.id/cuaca/pelabuhan/pelabuhan-probolinggo"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    parsed_data = []
    
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=12)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                text_content = soup.get_text()
                
                # Ekstrak pola nilai pasang surut (misal: +0.42m, -0.85m, +0.00m)
                # rePattern menyesuaikan dengan format render text BMKG
                matches = re.findall(r'([+-]\d+\.\d+)[\s]*m', text_content)
                
                if matches:
                    # Konversi string angka float ke nilai elevasi
                    elevations = [float(val) for val in matches]
                    
                    # Buat rentang waktu jam-demi-jam dari jam saat ini
                    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
                    time_series = [start_time + timedelta(hours=i) for i in range(len(elevations))]
                    
                    df = pd.DataFrame({
                        "Waktu": time_series,
                        "Elevasi (m)": elevations
                    })
                    return df, f"Berhasil terhubung ke: {url}"
        except Exception as e:
            continue

    return None, "Gagal menarik data dari BMKG Maritim. Silakan periksa koneksi internet."

# -------------------------------------------------------------------
# AMBIL DATA & RENDER DASHBOARD
# -------------------------------------------------------------------
st.sidebar.header("🔄 Kontrol Sistem")
if st.sidebar.button("Refresh Data BMKG Now"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Menghubungkan & Mengambil data langsung dari maritim.bmkg.go.id..."):
    df_bmkg, status_msg = get_bmkg_maritim_data()

st.sidebar.info(status_msg)

if df_bmkg is not None and not df_bmkg.empty:
    
    # Summary Ringkasan Metrics
    st.subheader("📊 Status Elevasi Air Saat Ini & Prediksi")
    
    current_val = df_bmkg.iloc[0]["Elevasi (m)"]
    max_val = df_bmkg["Elevasi (m)"].max()
    min_val = df_bmkg["Elevasi (m)"].min()
    mean_val = df_bmkg["Elevasi (m)"].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status Jam Ini", f"{current_val:+.2f} m")
    c2.metric("Pasang Maksimum (HWL)", f"{max_val:+.2f} m")
    c3.metric("Rata-rata Muka Air (MSL)", f"{mean_val:+.2f} m")
    c4.metric("Surut Minimum (LWL)", f"{min_val:+.2f} m")

    st.divider()

    # Visualisasi Plot Grafik
    st.subheader("📈 Grafik Tren Pasang Surut (Direct BMKG Data)")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_bmkg["Waktu"], df_bmkg["Elevasi (m)"], color="#0077B6", linewidth=2.5, marker="o", markersize=4, label="Data Pasut BMKG (m)")
    ax.axhline(mean_val, color="red", linestyle="--", alpha=0.7, label="Mean Sea Level (MSL)")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

    ax.set_ylabel("Ketinggian Muka Air (Meter)", fontsize=11)
    ax.set_xlabel("Waktu (WIB)", fontsize=11)
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # Tabel Data Rincian Jam-demi-Jam
    st.subheader("📋 Data Per Jam BMKG Maritim")
    df_display = df_bmkg.copy()
    df_display["Waktu"] = df_display["Waktu"].dt.strftime("%d %B %Y - %H:%M WIB")
    st.dataframe(df_display, use_container_width=True, height=400)

else:
    st.error("Gagal mendapatkan data pasang surut dari BMKG Maritim.")
    st.warning("Pastikan server Streamlit Anda memiliki akses outbound HTTP/HTTPS ke maritim.bmkg.go.id.")
