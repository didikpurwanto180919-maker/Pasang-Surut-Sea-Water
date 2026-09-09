import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Realtime Pasut Probolinggo / Pasuruan",
    page_icon="🌊",
    layout="wide"
)

# -------------------------------------------------------------------
# FITUR AUTO-REFRESH SISI CLIENT (60 DETIK)
# -------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    # Refresh otomatis halaman setiap 60.000 ms (60 detik)
    count = st_autorefresh(interval=60000, key="pasut_autorefresh")
except ImportError:
    # Fallback jika library streamlit-autorefresh belum terinstall
    st.sidebar.warning("Saran: Install `streamlit-autorefresh` via pip agar refresh lebih mulus.")

st.title("🌊 Real-time Pasang Surut - Probolinggo / Pasuruan")
st.caption("Data Pasang Surut Resmi Dishidros TNI-AL / BMKG (Chart Datum LWL)")

# Waktu Realtime Sistem Saat Ini (Jam:Menit:Detik)
now_time = datetime.now()

# -------------------------------------------------------------------
# DATA MATRIKS TABEL RESMI (SEPTEMBER 2026)
# -------------------------------------------------------------------
tide_table_data = {
    7: [2.1, 2.0, 2.0, 1.9, 2.0, 2.0, 2.0, 1.9, 1.8, 1.6, 1.3, 1.1, 0.8, 0.7, 0.6, 0.7, 0.9, 1.2, 1.5, 1.8, 2.0, 2.1, 2.1, 2.1],
    8: [1.7, 1.6, 1.6, 1.7, 1.9, 2.2, 2.4, 2.6, 2.5, 2.2, 1.8, 1.3, 0.8, 0.5, 0.2, 0.2, 0.5, 0.9, 1.4, 1.8, 2.2, 2.3, 2.2, 2.0],
    9: [1.8, 1.5, 1.4, 1.4, 1.6, 1.9, 2.3, 2.6, 2.7, 2.6, 2.3, 1.8, 1.2, 0.7, 0.3, 0.1, 0.2, 0.6, 1.1, 1.6, 2.1, 2.4, 2.4, 2.2],
    10: [1.9, 1.6, 1.3, 1.2, 1.3, 1.5, 1.9, 2.4, 2.7, 2.8, 2.6, 2.2, 1.6, 1.0, 0.5, 0.2, 0.2, 0.4, 0.8, 1.4, 1.9, 2.3, 2.5, 2.4],
    11: [2.1, 1.7, 1.3, 1.1, 1.0, 1.2, 1.6, 2.0, 2.5, 2.7, 2.7, 2.5, 2.0, 1.4, 0.9, 0.4, 0.3, 0.3, 0.7, 1.2, 1.7, 2.2, 2.5, 2.5]
}

def load_and_interpolate_data():
    times = []
    elevations = []
    
    # Susun Data Per Jam
    for day, vals in tide_table_data.items():
        for hour_idx, val in enumerate(vals):
            dt = datetime(2026, 9, day, hour_idx, 0, 0)
            times.append(dt)
            elevations.append(val)
            
    df = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevations})
    
    # Interpolasi Nilai Realtime Menit-demi-Menit untuk Waktu Sekarang
    df_sorted = df.sort_values("Waktu").reset_index(drop=True)
    
    # Hitung elevasi persis pada menit saat ini
    prev_row = df_sorted[df_sorted["Waktu"] <= now_time].iloc[-1] if not df_sorted[df_sorted["Waktu"] <= now_time].empty else df_sorted.iloc[0]
    next_row = df_sorted[df_sorted["Waktu"] > now_time].iloc[0] if not df_sorted[df_sorted["Waktu"] > now_time].empty else df_sorted.iloc[-1]
    
    if prev_row["Waktu"] == next_row["Waktu"]:
        current_elev = prev_row["Elevasi (m)"]
    else:
        # Interpolasi Linear Sesuai Menit
        time_diff = (next_row["Waktu"] - prev_row["Waktu"]).total_seconds()
        current_diff = (now_time - prev_row["Waktu"]).total_seconds()
        weight = current_diff / time_diff
        current_elev = prev_row["Elevasi (m)"] + weight * (next_row["Elevasi (m)"] - prev_row["Elevasi (m)"])
        
    return df_sorted, current_elev

df_tide, current_val = load_and_interpolate_data()

# -------------------------------------------------------------------
# METRIK BUKAN RETAIL
# -------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Muka Air Realtime ({now_time.strftime('%H:%M:%S WIB')})", f"{current_val:.2f} m")
c2.metric("Pasang Tertinggi (HWL)", f"{df_tide['Elevasi (m)'].max():.1f} m")
c3.metric("Rata-rata (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
c4.metric("Surut Terendah (LWL)", f"{df_tide['Elevasi (m)'].min():.1f} m")

st.caption(f"⚡ *Auto-refresh aktif. Terakhir diperbarui pada: {now_time.strftime('%d %B %Y - %H:%M:%S WIB')}*")
st.divider()

# -------------------------------------------------------------------
# GRAFIK MATPLOTLIB DINAMIS
# -------------------------------------------------------------------
st.subheader("📈 Grafik Elevasi Pasang Surut Realtime")

fig, ax = plt.subplots(figsize=(15, 6))

# Plot Kurva Utama
ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2.5, marker="o", markersize=3.5, label="Elevasi Chart Datum (m)")

# Garis Rata-rata (MSL)
msl_val = df_tide['Elevasi (m)'].mean()
ax.axhline(msl_val, color="red", linestyle="--", alpha=0.6, label=f"MSL ({msl_val:.2f} m)")

# Penanda Garis Vertikal & Titik REALTIME "SAAT INI"
ax.axvline(now_time, color="#D62728", linestyle="-", linewidth=2, label=f"Saat Ini ({now_time.strftime('%H:%M WIB')})")
ax.plot(now_time, current_val, marker="o", markersize=10, color="#D62728")

# Label Angka Teks di Setiap Titik Jam
for x, y in zip(df_tide["Waktu"], df_tide["Elevasi (m)"]):
    ax.annotate(
        f"{y:.1f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 7),
        ha='center',
        fontsize=7.5,
        fontweight='bold',
        color='#03045E'
    )

# Highlight Banner "SAAT INI" Mengikuti Menit & Jam Jalan
ax.annotate(
    f"SAAT INI: {current_val:.2f}m\n({now_time.strftime('%H:%M WIB')})",
    (now_time, current_val),
    textcoords="offset points",
    xytext=(0, -32),
    ha='center',
    fontsize=9,
    fontweight='bold',
    color='#D62728',
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFCC", ec="#D62728", lw=1.5, alpha=0.9)
)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

ax.set_ylim(-0.1, df_tide["Elevasi (m)"].max() + 0.4)
ax.set_ylabel("Ketinggian Muka Air / Chart Datum (Meter)", fontsize=11)
ax.set_xlabel("Waktu (WIB)", fontsize=11)
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
plt.xticks(rotation=35)
plt.tight_layout()

st.pyplot(fig)

# -------------------------------------------------------------------
# TABEL DETAIL
# -------------------------------------------------------------------
with st.expander("📄 Rincian Tabel Pasang Surut"):
    df_show = df_tide.copy()
    df_show["Waktu"] = df_show["Waktu"].dt.strftime("%d %B %Y - %H:%M WIB")
    st.dataframe(df_show, use_container_width=True)
