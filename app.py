import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Pasut Probolinggo / Pasuruan",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Real-time Pasang Surut - Probolinggo / Pasuruan")
st.caption("Referensi Data: **Tabel Pasang Surut Resmi Dishidros/BMKG (Chart Datum LWL)**")

# Waktu Sekarang (System Time)
now_time = datetime.now().replace(second=0, microsecond=0)

# -------------------------------------------------------------------
# DATA HARIAN DARI TABEL RESMI (SEPTEMBER 2026)
# -------------------------------------------------------------------
# Data Tabel Resmi Tanggal 8, 9, 10 September (Jam 1 s/d 24 WIB)
tide_table_data = {
    8: [1.7, 1.6, 1.6, 1.7, 1.9, 2.2, 2.4, 2.6, 2.5, 2.2, 1.8, 1.3, 0.8, 0.5, 0.2, 0.2, 0.5, 0.9, 1.4, 1.8, 2.2, 2.3, 2.2, 2.0],
    9: [1.8, 1.5, 1.4, 1.4, 1.6, 1.9, 2.3, 2.6, 2.7, 2.6, 2.3, 1.8, 1.2, 0.7, 0.3, 0.1, 0.2, 0.6, 1.1, 1.6, 2.1, 2.4, 2.4, 2.2],
    10: [1.9, 1.6, 1.3, 1.2, 1.3, 1.5, 1.9, 2.4, 2.7, 2.8, 2.6, 2.2, 1.6, 1.0, 0.5, 0.2, 0.2, 0.4, 0.8, 1.4, 1.9, 2.3, 2.5, 2.4]
}

def load_official_tide_data():
    times = []
    elevations = []
    
    # Ambil data tanggal 8 s/d 10 September 2026
    for day, vals in tide_table_data.items():
        for hour_idx, val in enumerate(vals):
            dt = datetime(2026, 9, day, hour_idx)
            times.append(dt)
            elevations.append(val)
            
    return pd.DataFrame({"Waktu": times, "Elevasi (m)": elevations})

# -------------------------------------------------------------------
# RENDER DASHBOARD
# -------------------------------------------------------------------
df_tide = load_official_tide_data()

# Cari titik waktu terdekat jam sekarang
df_tide['diff'] = abs(df_tide['Waktu'] - now_time)
current_row = df_tide.loc[df_tide['diff'].idxmin()]
current_val = current_row["Elevasi (m)"]

# Metrik Dashboard Utama
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Muka Air Saat Ini ({current_row['Waktu'].strftime('%H:00 WIB')})", f"{current_val:.1f} m")
c2.metric("Pasang Tertinggi (HWL)", f"{df_tide['Elevasi (m)'].max():.1f} m")
c3.metric("Rata-rata (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
c4.metric("Surut Terendah (LWL)", f"{df_tide['Elevasi (m)'].min():.1f} m")

st.divider()

# Grafik Pasang Surut
st.subheader("📈 Grafik Elevasi Pasang Surut (Sesuai Tabel Resmi Dishidros/BMKG)")

fig, ax = plt.subplots(figsize=(15, 6))

# Plot Kurva Utama
ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2.5, marker="o", markersize=4, label="Elevasi Air Laut Chart Datum (m)")

# Garis MSL
msl_val = df_tide['Elevasi (m)'].mean()
ax.axhline(msl_val, color="red", linestyle="--", alpha=0.6, label=f"MSL ({msl_val:.2f} m)")

# Penanda Jam Sekarang
ax.axvline(now_time, color="#D62728", linestyle="-", linewidth=2, label=f"Saat Ini ({now_time.strftime('%d-%b %H:%M')})")
ax.plot(current_row["Waktu"], current_val, marker="o", markersize=10, color="#D62728")

# Tampilkan angka di tiap titik data
for x, y in zip(df_tide["Waktu"], df_tide["Elevasi (m)"]):
    ax.annotate(
        f"{y:.1f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 7),
        ha='center',
        fontsize=8,
        fontweight='bold',
        color='#03045E'
    )

# Highlight Waktu Saat Ini
ax.annotate(
    f"SAAT INI: {current_val:.1f}m",
    (current_row["Waktu"], current_val),
    textcoords="offset points",
    xytext=(0, -22),
    ha='center',
    fontsize=9,
    fontweight='bold',
    color='#D62728',
    bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="#D62728", lw=1, alpha=0.8)
)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

ax.set_ylim(0, df_tide["Elevasi (m)"].max() + 0.4)
ax.set_ylabel("Ketinggian Muka Air / Chart Datum (Meter)", fontsize=11)
ax.set_xlabel("Waktu (WIB)", fontsize=11)
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
plt.xticks(rotation=40)
plt.tight_layout()

st.pyplot(fig)

# Tabel Data Rincian
with st.expander("📄 Lihat Rincian Tabel Per Jam"):
    df_show = df_tide.drop(columns=['diff']).copy()
    df_show["Waktu"] = df_show["Waktu"].dt.strftime("%d %B %Y - %H:%M WIB")
    st.dataframe(df_show, use_container_width=True)
