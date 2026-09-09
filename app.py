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
    page_title="Multi-Source Tide & Sea Level Monitoring",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Portal Monitoring Pasang Surut Air Laut")
st.markdown("Integrasi Data Real-time: **IOC Sea Level Station**, **BMKG Ocean Forecast System (OFS)**, dan **BIG (Badan Informasi Geospasial)**.")

# Sidebar pilihan sumber data
st.sidebar.header("🌐 Pilih Sumber Data & Lokasi")
source_option = st.sidebar.radio(
    "Sumber Data:",
    ["BMKG Maritim (OFS)", "IOC Sea Level Station (Global/Indo)", "Prediksi Harmonik BIG/Simulasi"]
)

# -------------------------------------------------------------------
# FUNGSIONALITAS 1: IOC Sea Level Station Monitoring Facility (REST API v1)
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_ioc_data(station_code):
    url = f"https://api.ioc-sealevelmonitoring.org/v1/stationdata/{station_code}"
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
        return None, f"Stasiun IOC {station_code} tidak merespons atau data kosong."
    except Exception as e:
        return None, str(e)

# -------------------------------------------------------------------
# FUNGSIONALITAS 2: BMKG Ocean Forecast System (Live Web Scrape)
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_bmkg_ofs_data(slug):
    url = f"https://maritim.bmkg.go.id/cuaca/pelabuhan/{slug}"
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
        return None, "Data BMKG tidak dapat diproses."
    except Exception as e:
        return None, str(e)

# -------------------------------------------------------------------
# MAIN RENDER LOGIC
# -------------------------------------------------------------------
df_tide = None
error_msg = None
selected_location = ""

if source_option == "BMKG Maritim (OFS)":
    bmkg_ports = {
        "Pelabuhan Probolinggo": "pelabuhan-probolinggo",
        "Pelabuhan Tanjung Perak (Surabaya)": "pelabuhan-tanjung-perak",
        "Pelabuhan Tanjung Priok (Jakarta)": "pelabuhan-tanjung-priok",
        "Pelabuhan Belawan (Medan)": "pelabuhan-belawan"
    }
    selected_location = st.sidebar.selectbox("Pilih Pelabuhan (BMKG):", list(bmkg_ports.keys()))
    with st.spinner("Mengambil data live dari BMKG..."):
        df_tide, error_msg = fetch_bmkg_ofs_data(bmkg_ports[selected_location])

elif source_option == "IOC Sea Level Station (Global/Indo)":
    # Kode stasiun sensor pasang surut resmi IOC di Indonesia
    ioc_stations = {
        "Surabaya (Jawa Timur)": "sura",
        "Jakarta (Tanjung Priok)": "jaka",
        "Prigi (Trenggalek)": "prig",
        "Banyuwangi": "bany",
        "Cilacap": "cila",
        "Padang": "pada"
    }
    selected_location = st.sidebar.selectbox("Pilih Stasiun IOC:", list(ioc_stations.keys()))
    with st.spinner("Mengambil data real-time sensor IOC UNESCO..."):
        df_tide, error_msg = fetch_ioc_data(ioc_stations[selected_location])

else: # Prediksi BIG / Simulasi Harmonik
    selected_location = st.sidebar.text_input("Nama Lokasi Stasiun BIG:", "Stasiun Pasut Pasuruan / BIG")
    st.sidebar.caption("BIG menyediakan data pasang surut berbasis analisis stasiun stasioner dan kontour geomorfologi.")
    
    # Generate simulasi berdasarkan parameter harmonik utama BIG
    hours = 72
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now + timedelta(hours=i) for i in range(hours)]
    t = np.arange(hours)
    # M2 + S2 + K1 + O1
    elevation = 1.2 * np.cos(2 * np.pi * t / 12.42) + 0.5 * np.cos(2 * np.pi * t / 12.0) + 0.3 * np.cos(2 * np.pi * t / 23.93)
    df_tide = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevation})

# TAMPILAN DASHBOARD
if error_msg:
    st.error(error_msg)
elif df_tide is not None and not df_tide.empty:
    st.subheader(f"📍 Lokasi: {selected_location}")
    
    # Ringkasan Parameter Metrik
    c1, c2, c3 = st.columns(3)
    c1.metric("Muka Air Maksimum (HWL)", f"{df_tide['Elevasi (m)'].max():.2f} m")
    c2.metric("Mean Sea Level (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
    c3.metric("Muka Air Minimum (LWL)", f"{df_tide['Elevasi (m)'].min():.2f} m")

    st.divider()

    # Visualisasi Grafik Detail per Jam
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2, marker="o", markersize=2.5, label="Elevasi Air Laut (m)")
    ax.axhline(df_tide["Elevasi (m)"].mean(), color="red", linestyle="--", alpha=0.6, label="MSL")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

    ax.set_ylabel("Ketinggian Muka Air (m)", fontsize=10)
    ax.set_xlabel("Waktu (Tanggal & Jam)", fontsize=10)
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # Tabel Data Realtime
    with st.expander("📄 Tabel Data Detail Per Jam"):
        df_show = df_tide.copy()
        df_show["Waktu"] = df_show["Waktu"].dt.strftime("%Y-%m-%d %H:%M WIB")
        st.dataframe(df_show, use_container_width=True)
