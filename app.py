import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Prediksi Pasang Surut Air Laut",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Aplikasi Prediksi Pasang Surut Air Laut")
st.markdown("Simulasi dan kalkulasi elevasi pasang surut berdasarkan komponen harmonik.")

# Sidebar - Parameter Input
st.sidebar.header("⚙️ Parameter Pasang Surut")

darat_mean = st.sidebar.number_input("Mean Sea Level / MSL (m)", value=1.5, step=0.1)
amp_m2 = st.sidebar.number_input("Amplitudo M2 - Utama Bulan (m)", value=0.8, step=0.1)
amp_s2 = st.sidebar.number_input("Amplitudo S2 - Utama Matahari (m)", value=0.4, step=0.1)
amp_k1 = st.sidebar.number_input("Amplitudo K1 - Deklinasi Tunggal (m)", value=0.3, step=0.1)

jam_simulasi = st.sidebar.slider("Durasi Simulasi (Jam)", min_value=24, max_value=168, value=72, step=24)

# Kalkulasi Pasang Surut (Komponen Harmonik Sederhana)
@st.cache_data
def generate_tide_data(msl, m2, s2, k1, hours):
    now = datetime.now()
    time_series = [now + timedelta(hours=i) for i in range(hours)]
    t = np.arange(hours)
    
    # Periode komponen pasang surut (dalam jam)
    T_M2, T_S2, T_K1 = 12.42, 12.00, 23.93
    
    # Formula elevasi muka air: Y(t) = MSL + M2 + S2 + K1
    elevation = (
        msl +
        m2 * np.cos(2 * np.pi * t / T_M2) +
        s2 * np.cos(2 * np.pi * t / T_S2) +
        k1 * np.cos(2 * np.pi * t / T_K1)
    )
    
    df = pd.DataFrame({"Waktu": time_series, "Elevasi (m)": elevation})
    return df

data = generate_tide_data(darat_mean, amp_m2, amp_s2, amp_k1, jam_simulasi)

# Ringkasan Parameter Ringkas
col1, col2, col3 = st.columns(3)
col1.metric("Muka Air Maksimum (HWL)", f"{data['Elevasi (m)'].max():.2f} m")
col2.metric("Rata-rata Muka Air (MSL)", f"{data['Elevasi (m)'].mean():.2f} m")
col3.metric("Muka Air Minimum (LWL)", f"{data['Elevasi (m)'].min():.2f} m")

st.divider()

# Grafik Hasil Prediksi
st.subheader("📈 Grafik Elevasi Muka Air")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(data["Waktu"], data["Elevasi (m)"], color="#0077B6", linewidth=2, label="Muka Air Air Laut")
ax.axhline(darat_mean, color="red", linestyle="--", alpha=0.7, label="MSL (Mean Sea Level)")
ax.set_ylabel("Elevasi (Meter)")
ax.set_xlabel("Waktu")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")
plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig)

# Data Tabel
with st.expander("📄 Lihat Data Mentah Simulasi"):
    st.dataframe(data, use_container_width=True)
