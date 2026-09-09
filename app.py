import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime, timedelta

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="BMKG Maritim - Pelabuhan Probolinggo",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Real-Time & Forecast Pasang Surut - Pelabuhan Probolinggo")
st.caption("Direct Data Stream dari BMKG Maritim: `maritim.bmkg.go.id/cuaca/pelabuhan/pelabuhan-probolinggo`")

# -------------------------------------------------------------------
# AMBIL DATA DARI ENDPOINT API / WEB BMKG MARITIM
# -------------------------------------------------------------------
@st.cache_data(ttl=900)  # Refresh cache otomatis setiap 15 menit
def fetch_probolinggo_direct():
    # URL Endpoint API JSON BMKG Maritim & URL Publik
    api_url = "https://maritim.bmkg.go.id/api/pelabuhan/pelabuhan-probolinggo"
    web_url = "https://maritim.bmkg.go.id/cuaca/pelabuhan/pelabuhan-probolinggo"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": web_url
    }
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        
        # Jika Endpoint API Merespons JSON
        if response.status_code == 200:
            json_data = response.json()
            
            # Parsing payload pasang surut dari JSON BMKG
            if "data" in json_data and "tide" in json_data["data"]:
                tide_list = json_data["data"]["tide"]
                
                times = []
                elevations = []
                
                for item in tide_list:
                    # Ambil waktu dan nilai elevasi pasang surut
                    times.append(pd.to_datetime(item["time"]))
                    elevations.append(float(item["value"]))
                
                df = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevations})
                return df, "Berhasil terhubung langsung via API BMKG Maritim JSON."

    except Exception:
        pass

    # Fallback Scraper jika API memerlukan token/session khusus
    try:
        res = requests.get(web_url, headers=headers, timeout=10)
        if res.status_code == 200:
            from bs4 import BeautifulSoup
            import re
            
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text()
            matches = re.findall(r'([+-]\d+\.\d+)[\s]*m', text)
            
            if matches:
                elevations = [float(val) for val in matches]
                start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
                times = [start_time + timedelta(hours=i) for i in range(len(elevations))]
                df = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevations})
                return df, "Berhasil terhubung via Scraper Web BMKG Maritim."
    except Exception as e:
        return None, f"Gagal mengambil data dari server BMKG: {str(e)}"

    return None, "Gagal mengurai respons data pasang surut BMKG Probolinggo."

# -------------------------------------------------------------------
# DASHBOARD DISPLAY
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Kontrol")
if st.sidebar.button("🔄 Refresh Data Realtime"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Menghubungkan ke maritim.bmkg.go.id..."):
    df_tide, status_msg = fetch_probolinggo_direct()

st.sidebar.info(status_msg)

if df_tide is not None and not df_tide.empty:
    
    # Kumpulan Indikator Utama
    hwl = df_tide["Elevasi (m)"].max()
    msl = df_tide["Elevasi (m)"].mean()
    lwl = df_tide["Elevasi (m)"].min()
    current = df_tide.iloc[0]["Elevasi (m)"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Muka Air Saat Ini", f"{current:+.2f} m")
    c2.metric("Pasang Tertinggi (HWL)", f"{hwl:+.2f} m")
    c3.metric("Mean Sea Level (MSL)", f"{msl:+.2f} m")
    c4.metric("Surut Terendah (LWL)", f"{lwl:+.2f} m")

    st.divider()

    # Grafik Detail Jam-demi-Jam
    st.subheader("📈 Grafik Elevasi Pasang Surut Per Jam (WIB)")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2.5, marker="o", markersize=3.5, label="Elevasi Air Laut BMKG (m)")
    ax.axhline(msl, color="red", linestyle="--", alpha=0.7, label=f"MSL ({msl:.2f} m)")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

    ax.set_ylabel("Ketinggian Muka Air (Meter)", fontsize=11)
    ax.set_xlabel("Waktu (Tanggal & Jam WIB)", fontsize=11)
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # Tabel Data Rincian
    with st.expander("📄 Lihat Rincian Tabel Per Jam"):
        df_table = df_tide.copy()
        df_table["Waktu"] = df_table["Waktu"].dt.strftime("%Y-%m-%d %H:%M WIB")
        st.dataframe(df_table, use_container_width=True)

else:
    st.error("Data tidak dapat dimuat dari BMKG Maritim.")
    st.warning("Silakan klik tombol 'Refresh Data Realtime' pada sidebar.")
