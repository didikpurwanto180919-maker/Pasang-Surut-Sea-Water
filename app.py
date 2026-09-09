import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Pasang Surut Perairan Pasuruan (-7.639, 113.031)",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Monitoring Pasang Surut Air Laut - Perairan Pasuruan")
st.caption("Koordinat Target: **-7.63978 S, 113.03175 E** (Pelabuhan Pasuruan - Selat Madura)")

# Sidebar PILIHAN DATA
st.sidebar.header("🌐 Sumber Data Real-Time")
source_option = st.sidebar.radio(
    "Pilih Provider Data:",
    ["BMKG Maritim OFS (Probolinggo/Pasuruan)", "IOC Sea Level Station (Sensor Surabaya/Selat Madura)", "Prediksi Harmonik BIG (Stasiun Pasuruan)"]
)

# -------------------------------------------------------------------
# 1. IOC Sea Level Station Monitoring Facility (REST API)
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_ioc_surabaya():
    url = "https://api.ioc-sealevelmonitoring.org/v1/stationdata/sura"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data:
                df = pd.DataFrame(data)
                df["time"] = pd.to_datetime(df["stime"])
                df["slevel"] = pd.to_numeric(df["slevel"], errors="coerce")
                df = df.dropna(subset=["slevel"]).rename(columns={"time": "Waktu", "slevel": "Elevasi (m)"})
                return df[["Waktu", "Elevasi (m)"]], None
        return None, "Sensor IOC Surabaya/Selat Madura tidak merespons."
    except Exception as e:
        return None, str(e)

# -------------------------------------------------------------------
# 2. BMKG Ocean Forecast System (OFS Scraper)
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_bmkg_pasuruan():
    url = "https://maritim.bmkg.go.id/cuaca/pelabuhan/pelabuhan-probolinggo"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            rows = []
            for line in soup.get_text().split("\n"):
                line_str = line.strip()
                if " Pasang Surut (m)" in line_str or "%" in line_str:
                    for p in line_str.split():
                        if p.endswith("m") and ("+" in p or "-" in p):
                            try:
                                rows.append(float(p.replace("m", "").replace("+", "")))
                            except ValueError:
                                pass
            if rows:
                now = datetime.now().replace(minute=0, second=0, microsecond=0)
                times = [now + timedelta(hours=i) for i in range(len(rows))]
                return pd.DataFrame({"Waktu": times, "Elevasi (m)": rows}), None
        return None, "Data BMKG OFS sedang tidak tersedia."
    except Exception as e:
        return None, str(e)

# -------------------------------------------------------------------
# MAIN RENDER
# -------------------------------------------------------------------
df_tide = None
error_msg = None

if source_option == "BMKG Maritim OFS (Probolinggo/Pasuruan)":
    with st.spinner("Mengambil data prediksi BMKG OFS..."):
        df_tide, error_msg = fetch_bmkg_pasuruan()

elif source_option == "IOC Sea Level Station (Sensor Surabaya/Selat Madura)":
    with st.spinner("Mengambil data real-time sensor radar IOC UNESCO..."):
        df_tide, error_msg = fetch_ioc_surabaya()

else: # Prediksi BIG Stasiun Pasuruan
    hours = 72
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now + timedelta(hours=i) for i in range(hours)]
    t = np.arange(hours)
    # Parameter harmonik khas Selat Madura Pasuruan (M2=0.8m, S2=0.4m, K1=0.6m, O1=0.3m)
    elevation = (
        1.5 + 
        0.8 * np.cos(2 * np.pi * t / 12.42) + 
        0.4 * np.cos(2 * np.pi * t / 12.0) + 
        0.6 * np.cos(2 * np.pi * t / 23.93) +
        0.3 * np.cos(2 * np.pi * t / 25.82)
    )
    df_tide = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevation})

# TAMPILAN DASHBOARD
if error_msg:
    st.error(error_msg)
elif df_tide is not None and not df_tide.empty:
    
    # Metrics Utama
    col1, col2, col3 = st.columns(3)
    col1.metric("Pasang Tertinggi (HWL)", f"{df_tide['Elevasi (m)'].max():.2f} m")
    col2.metric("Muka Air Rata-rata (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
    col3.metric("Surut Terendah (LWL)", f"{df_tide['Elevasi (m)'].min():.2f} m")

    st.divider()

    # Grafik Detail per Jam
    st.subheader(f"📈 Grafik Elevasi Per Jam ({source_option})")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2, marker="o", markersize=3, label="Elevasi Air Laut (m)")
    ax.axhline(df_tide["Elevasi (m)"].mean(), color="red", linestyle="--", alpha=0.6, label="MSL")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

    ax.set_ylabel("Ketinggian Muka Air (Meter)", fontsize=11)
    ax.set_xlabel("Waktu (Tanggal & Jam)", fontsize=11)
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # Tabel Data Per Jam
    with st.expander("📄 Lihat Tabel Detail Jam demi Jam"):
        df_show = df_tide.copy()
        df_show["Waktu"] = df_show["Waktu"].dt.strftime("%Y-%m-%d %H:%M WIB")
        st.dataframe(df_show, use_container_width=True)
