import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz

# Page Configuration
st.set_page_config(
    page_title="Monitoring Pasut Realtime - GPS S07°38.659' E113°01.641'",
    page_icon="🌊",
    layout="wide"
)

# Timezone Setup
wib_tz = pytz.timezone('Asia/Jakarta')
now_time = datetime.now(wib_tz).replace(tzinfo=None)

# Auto-refresh every 60 seconds
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="pasut_autorefresh_gps")
except ImportError:
    pass

st.title("🌊 Sistem Monitoring Pasang Surut Realtime")

st.markdown("""
> 📍 **Lokasi Pemantau (GPS):** `S 07° 38.659' E 113° 01.641'`  
> 🗺️ **Wilayah:** Pesisir Selat Madura (Pasuruan - Probolinggo)  
> 📊 **Acuan Datum:** Chart Datum Low Water Level (LWL) - Standard Dishidros TNI-AL / BMKG  
""")

# -------------------------------------------------------------------
# DATA MATRIKS PASUT TERVALIDASI DARI TABEL DISHIDROS
# -------------------------------------------------------------------
tide_matrix_dishidros = {
    8:  [1.7, 1.6, 1.6, 1.7, 1.9, 2.2, 2.4, 2.6, 2.5, 2.2, 1.8, 1.3, 0.8, 0.5, 0.2, 0.2, 0.5, 0.9, 1.4, 1.8, 2.2, 2.3, 2.2, 2.0],
    9:  [1.8, 1.5, 1.4, 1.4, 1.6, 1.9, 2.3, 2.6, 2.7, 2.6, 2.3, 1.8, 1.2, 0.7, 0.3, 0.1, 0.2, 0.6, 1.1, 1.6, 2.1, 2.4, 2.4, 2.2],
    10: [1.9, 1.6, 1.3, 1.2, 1.3, 1.5, 1.9, 2.4, 2.7, 2.8, 2.6, 2.2, 1.6, 1.0, 0.5, 0.2, 0.2, 0.4, 0.8, 1.4, 1.9, 2.3, 2.5, 2.4]
}

@st.cache_data
def build_base_dataframe():
    times = []
    elevations = []
    
    for day, row_vals in tide_matrix_dishidros.items():
        base_dt = datetime(2026, 9, day, 0, 0, 0)
        for col_idx, val in enumerate(row_vals):
            dt = base_dt + timedelta(hours=col_idx)
            times.append(dt)
            elevations.append(val)
            
    df = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevations}).sort_values("Waktu").reset_index(drop=True)
    return df

def get_realtime_tide_data(df, current_dt):
    # Safely bound lookup within available data range
    if current_dt <= df["Waktu"].iloc[0]:
        return df.iloc[0]["Elevasi (m)"], df.iloc[0], df.iloc[0]
    elif current_dt >= df["Waktu"].iloc[-1]:
        return df.iloc[-1]["Elevasi (m)"], df.iloc[-1], df.iloc[-1]
    
    prev_r = df[df["Waktu"] <= current_dt].iloc[-1]
    next_r = df[df["Waktu"] > current_dt].iloc[0]
    
    if prev_r["Waktu"] == next_r["Waktu"]:
        cur_elev = prev_r["Elevasi (m)"]
    else:
        t_diff = (next_r["Waktu"] - prev_r["Waktu"]).total_seconds()
        c_diff = (current_dt - prev_r["Waktu"]).total_seconds()
        cur_elev = prev_r["Elevasi (m)"] + (c_diff / t_diff) * (next_r["Elevasi (m)"] - prev_r["Elevasi (m)"])
        
    return cur_elev, prev_r, next_r

df_tide = build_base_dataframe()
current_val, prev_point, next_point = get_realtime_tide_data(df_tide, now_time)

# -------------------------------------------------------------------
# METRICS DISPLAY
# -------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    label=f"Muka Air GPS ({now_time.strftime('%H:%M:%S WIB')})", 
    value=f"{current_val:.2f} m"
)
c2.metric("Pasang Tertinggi (HWL)", f"{df_tide['Elevasi (m)'].max():.1f} m")
c3.metric("Rata-Rata Muka Air (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
c4.metric("Surut Terendah (LWL)", f"{df_tide['Elevasi (m)'].min():.1f} m")

st.caption(f"⚡ *Update otomatis aktif. Timestamp lokal: {now_time.strftime('%d %B %Y - %H:%M:%S WIB')}*")
st.divider()

# -------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------
st.subheader("📈 Kurva Elevasi Air Realtime Sesuai Koordinat Pemantauan")

fig, ax = plt.subplots(figsize=(14, 5.5))

# Plot tide curve
ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2.2, marker="o", markersize=3.5, label="Elevasi Air (Chart Datum / m)")

# Mean Sea Level Line
msl_val = df_tide['Elevasi (m)'].mean()
ax.axhline(msl_val, color="red", linestyle="--", alpha=0.6, label=f"MSL ({msl_val:.2f} m)")

# Current Time Marking
ax.axvline(now_time, color="#D62728", linestyle="-", linewidth=2, label=f"Waktu Sekarang ({now_time.strftime('%H:%M WIB')})")
ax.plot(now_time, current_val, marker="o", markersize=9, color="#D62728")

# Hourly Data Point Annotations
for x, y in zip(df_tide["Waktu"], df_tide["Elevasi (m)"]):
    ax.annotate(
        f"{y:.1f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 6),
        ha='center',
        fontsize=8,
        fontweight='bold',
        color='#03045E'
    )

# Realtime Marker Tag
ax.annotate(
    f"KOORDINAT GPS\nElevasi: {current_val:.2f} m\n({now_time.strftime('%H:%M WIB')})",
    (now_time, current_val),
    textcoords="offset points",
    xytext=(0, -38),
    ha='center',
    fontsize=8.5,
    fontweight='bold',
    color='#D62728',
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFCC", ec="#D62728", lw=1.5, alpha=0.9)
)

# Axis & Grid Formatting
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

ax.set_xlim(datetime(2026, 9, 8, 0, 0), datetime(2026, 9, 10, 23, 59))
ax.set_ylim(-0.1, df_tide["Elevasi (m)"].max() + 0.4)

ax.set_ylabel("Ketinggian Muka Air / Chart Datum (m)", fontsize=10)
ax.set_xlabel("Waktu (WIB)", fontsize=10)
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)  # Prevents memory accumulation

# Inspection Expander
with st.expander("🔍 Detail Rincian Interpolasi Data Saat Ini"):
    st.write(f"- **Titik Jam Sebelumnya ({prev_point['Waktu'].strftime('%H:%M WIB')}):** {prev_point['Elevasi (m)']} m")
    st.write(f"- **Titik Jam Berikutnya ({next_point['Waktu'].strftime('%H:%M WIB')}):** {next_point['Elevasi (m)']} m")
    st.write(f"- **Hasil Interpolasi ({now_time.strftime('%H:%M WIB')}):** `{current_val:.2f} m`")
