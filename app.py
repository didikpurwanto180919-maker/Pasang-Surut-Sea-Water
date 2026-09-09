import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Pasut Probolinggo / Pasuruan WIB",
    page_icon="🌊",
    layout="wide"
)

# -------------------------------------------------------------------
# SETTING ZONA WAKTU INDONESIA BARAT (WIB / GMT+7)
# -------------------------------------------------------------------
wib_tz = pytz.timezone('Asia/Jakarta')
# Ambil jam sekarang dalam zona waktu WIB murni
now_time = datetime.now(wib_tz).replace(tzinfo=None)

# Auto-Refresh 60 Detik
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="pasut_autorefresh")
except ImportError:
    pass

st.title("🌊 Real-time Pasang Surut - Probolinggo / Pasuruan")
st.caption("Koordinat: **S7°38.659' E113°01.641'** | Data Chart Datum LWL Dishidros/BMKG")

# -------------------------------------------------------------------
# DATA TABEL SEPTEMBER 2026
# -------------------------------------------------------------------
tide_table_data = {
    8: [1.7, 1.6, 1.6, 1.7, 1.9, 2.2, 2.4, 2.6, 2.5, 2.2, 1.8, 1.3, 0.8, 0.5, 0.2, 0.2, 0.5, 0.9, 1.4, 1.8, 2.2, 2.3, 2.2, 2.0],
    9: [1.8, 1.5, 1.4, 1.4, 1.6, 1.9, 2.3, 2.6, 2.7, 2.6, 2.3, 1.8, 1.2, 0.7, 0.3, 0.1, 0.2, 0.6, 1.1, 1.6, 2.1, 2.4, 2.4, 2.2],
    10: [1.9, 1.6, 1.3, 1.2, 1.3, 1.5, 1.9, 2.4, 2.7, 2.8, 2.6, 2.2, 1.6, 1.0, 0.5, 0.2, 0.2, 0.4, 0.8, 1.4, 1.9, 2.3, 2.5, 2.4]
}

def load_data():
    times = []
    elevations = []
    for day, vals in tide_table_data.items():
        for hour_idx, val in enumerate(vals):
            times.append(datetime(2026, 9, day, hour_idx, 0, 0))
            elevations.append(val)
            
    df = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevations}).sort_values("Waktu").reset_index(drop=True)
    
    # Hitung nilai interpolasi persis di menit berjalan jam WIB
    prev_r = df[df["Waktu"] <= now_time].iloc[-1] if not df[df["Waktu"] <= now_time].empty else df.iloc[0]
    next_r = df[df["Waktu"] > now_time].iloc[0] if not df[df["Waktu"] > now_time].empty else df.iloc[-1]
    
    if prev_r["Waktu"] == next_r["Waktu"]:
        cur_elev = prev_r["Elevasi (m)"]
    else:
        t_diff = (next_r["Waktu"] - prev_r["Waktu"]).total_seconds()
        c_diff = (now_time - prev_r["Waktu"]).total_seconds()
        cur_elev = prev_r["Elevasi (m)"] + (c_diff / t_diff) * (next_r["Elevasi (m)"] - prev_r["Elevasi (m)"])
        
    return df, cur_elev

df_tide, current_val = load_data()

# -------------------------------------------------------------------
# METRIK UTAMA
# -------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Muka Air WIB ({now_time.strftime('%H:%M:%S WIB')})", f"{current_val:.2f} m")
c2.metric("Pasang Tertinggi (HWL)", f"{df_tide['Elevasi (m)'].max():.1f} m")
c3.metric("Rata-rata (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
c4.metric("Surut Terendah (LWL)", f"{df_tide['Elevasi (m)'].min():.1f} m")

st.divider()

# -------------------------------------------------------------------
# GRAFIK MATPLOTLIB
# -------------------------------------------------------------------
st.subheader("📈 Grafik Elevasi Pasang Surut Realtime WIB")

fig, ax = plt.subplots(figsize=(15, 6))

# Plot Kurva utama
ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2.5, marker="o", markersize=3.5, label="Elevasi Chart Datum (m)")

# Garis MSL
msl_val = df_tide['Elevasi (m)'].mean()
ax.axhline(msl_val, color="red", linestyle="--", alpha=0.6, label=f"MSL ({msl_val:.2f} m)")

# Penanda Garis Vertikal jam 17:21 WIB
ax.axvline(now_time, color="#D62728", linestyle="-", linewidth=2, label=f"Saat Ini ({now_time.strftime('%H:%M WIB')})")
ax.plot(now_time, current_val, marker="o", markersize=10, color="#D62728")

# Tampilkan angka di tiap titik jam
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

# Highlight Banner "SAAT INI"
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

# Format Sumbu X & Y
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

# Batasi rentang sumbu X agar fokus pada tanggal 8-10 September saja
ax.set_xlim(datetime(2026, 9, 8, 0, 0), datetime(2026, 9, 10, 23, 59))
ax.set_ylim(-0.1, df_tide["Elevasi (m)"].max() + 0.4)

ax.set_ylabel("Ketinggian Muka Air / Chart Datum (Meter)", fontsize=11)
ax.set_xlabel("Waktu (WIB)", fontsize=11)
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
plt.xticks(rotation=35)
plt.tight_layout()

st.pyplot(fig)
