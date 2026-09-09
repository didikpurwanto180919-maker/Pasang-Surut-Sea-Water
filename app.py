import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz

st.set_page_config(
    page_title="Pasut Probolinggo / Pasuruan",
    page_icon="🌊",
    layout="wide"
)

# Set Zona Waktu Indonesia Barat (WIB)
wib_tz = pytz.timezone('Asia/Jakarta')
now_time = datetime.now(wib_tz).replace(tzinfo=None)

# Auto-refresh tiap 60 detik
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="pasut_autorefresh")
except ImportError:
    pass

st.title("🌊 Real-time Pasang Surut - Probolinggo / Pasuruan")
st.caption("Referensi Matriks Data: **Tabel Cetak Resmi Dishidros TNI-AL / BMKG (Chart Datum LWL)**")

# -------------------------------------------------------------------
# DATA MATRIKS RESMI TABEL DISHIDROS (SEPTEMBER 2026)
# Format Array: [Jam 01:00, Jam 02:00, ..., Jam 24:00 (00:00)]
# -------------------------------------------------------------------
tide_matrix_dishidros = {
    8:  [1.7, 1.6, 1.6, 1.7, 1.9, 2.2, 2.4, 2.6, 2.5, 2.2, 1.8, 1.3, 0.8, 0.5, 0.2, 0.2, 0.5, 0.9, 1.4, 1.8, 2.2, 2.3, 2.2, 2.0],
    9:  [1.8, 1.5, 1.4, 1.4, 1.6, 1.9, 2.3, 2.6, 2.7, 2.6, 2.3, 1.8, 1.2, 0.7, 0.3, 0.1, 0.2, 0.6, 1.1, 1.6, 2.1, 2.4, 2.4, 2.2],
    10: [1.9, 1.6, 1.3, 1.2, 1.3, 1.5, 1.9, 2.4, 2.7, 2.8, 2.6, 2.2, 1.6, 1.0, 0.5, 0.2, 0.2, 0.4, 0.8, 1.4, 1.9, 2.3, 2.5, 2.4]
}

def build_realtime_dataframe():
    times = []
    elevations = []
    
    for day, row_vals in tide_matrix_dishidros.items():
        for col_idx, val in enumerate(row_vals):
            # col_idx 0 = Jam 00:00 WIB, col_idx 16 = Jam 16:00 WIB, col_idx 17 = Jam 17:00 WIB
            dt = datetime(2026, 9, day, col_idx, 0, 0)
            times.append(dt)
            elevations.append(val)
            
    df = pd.DataFrame({"Waktu": times, "Elevasi (m)": elevations}).sort_values("Waktu").reset_index(drop=True)
    
    # Hitung interpolasi realtime sesuai menit jam sekarang
    prev_r = df[df["Waktu"] <= now_time].iloc[-1] if not df[df["Waktu"] <= now_time].empty else df.iloc[0]
    next_r = df[df["Waktu"] > now_time].iloc[0] if not df[df["Waktu"] > now_time].empty else df.iloc[-1]
    
    if prev_r["Waktu"] == next_r["Waktu"]:
        cur_elev = prev_r["Elevasi (m)"]
    else:
        t_diff = (next_r["Waktu"] - prev_r["Waktu"]).total_seconds()
        c_diff = (now_time - prev_r["Waktu"]).total_seconds()
        cur_elev = prev_r["Elevasi (m)"] + (c_diff / t_diff) * (next_r["Elevasi (m)"] - prev_r["Elevasi (m)"])
        
    return df, cur_elev

df_tide, current_val = build_realtime_dataframe()

# -------------------------------------------------------------------
# DISPLAY METRIK REALTIME
# -------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Muka Air Realtime ({now_time.strftime('%H:%M:%S WIB')})", f"{current_val:.2f} m")
c2.metric("Pasang Tertinggi (HWL)", f"{df_tide['Elevasi (m)'].max():.1f} m")
c3.metric("Rata-rata Muka Air (MSL)", f"{df_tide['Elevasi (m)'].mean():.2f} m")
c4.metric("Surut Terendah (LWL)", f"{df_tide['Elevasi (m)'].min():.1f} m")

st.caption(f"⚡ *Auto-refresh aktif. Terakhir diperbarui: {now_time.strftime('%d %B %Y - %H:%M:%S WIB')}*")
st.divider()

# -------------------------------------------------------------------
# GRAFIK MATPLOTLIB
# -------------------------------------------------------------------
st.subheader("📈 Grafik Elevasi Pasang Surut Sesuai Tabel Resmi")

fig, ax = plt.subplots(figsize=(15, 6))

# Plot Kurva utama
ax.plot(df_tide["Waktu"], df_tide["Elevasi (m)"], color="#0077B6", linewidth=2.5, marker="o", markersize=4, label="Elevasi Chart Datum (m)")

# Garis MSL
msl_val = df_tide['Elevasi (m)'].mean()
ax.axhline(msl_val, color="red", linestyle="--", alpha=0.6, label=f"MSL ({msl_val:.2f} m)")

# Penanda Garis Vertikal & Titik Jam Sekarang (misal 17:23 WIB)
ax.axvline(now_time, color="#D62728", linestyle="-", linewidth=2, label=f"Saat Ini ({now_time.strftime('%H:%M WIB')})")
ax.plot(now_time, current_val, marker="o", markersize=10, color="#D62728")

# Tampilkan Angka di Setiap Titik Jam
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

# Batasi Sumbu X untuk Tanggal 8 s/d 10 September
ax.set_xlim(datetime(2026, 9, 8, 0, 0), datetime(2026, 9, 10, 23, 59))
ax.set_ylim(-0.1, df_tide["Elevasi (m)"].max() + 0.4)

ax.set_ylabel("Ketinggian Muka Air / Chart Datum (Meter)", fontsize=11)
ax.set_xlabel("Waktu (WIB)", fontsize=11)
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
plt.xticks(rotation=35)
plt.tight_layout()

st.pyplot(fig)
