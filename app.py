import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Prediksi Pasang Surut Air Laut Probolinggo",
    page_icon="🌊",
    layout="wide"
)

# ==========================================
# 2. FUNGSI MEMBACA DATA EXCEL
# ==========================================
@st.cache_data
def load_data(file_path, sheet_name):
    """Membaca sheet Excel berdasarkan nama bulan."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"Gagal membaca sheet '{sheet_name}': {e}")
        return None

# ==========================================
# 3. HEADER & TAMPILAN UTAMA
# ==========================================
st.title("🌊 Aplikasi Prediksi Pasang Surut Air Laut")
st.markdown("Stasiun **Probolinggo** | Lintang `07° 44' 10.79\" S` | Bujur `113° 12' 59.64\" T` (GMT +07:00)")

# ==========================================
# 4. SIDEBAR (FILTER INPUT)
# ==========================================
st.sidebar.header("⚙️ Filter Data")

# Nama file Excel (sesuaikan dengan lokasi file Anda)
FILE_EXCEL = "Tabel_Pasang_Surut_Probolinggo_2026.xlsx"

DAFTAR_BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", 
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

# Pilihan bulan dan tanggal di sidebar
bulan_pilihan = st.sidebar.selectbox("Pilih Bulan", DAFTAR_BULAN, index=8)  # Default: September
tanggal_pilihan = st.sidebar.number_input("Pilih Tanggal", min_value=1, max_value=31, value=9)

# ==========================================
# 5. PEMROSESAN DATA
# ==========================================
df = load_data(FILE_EXCEL, bulan_pilihan)

if df is not None:
    # Identifikasi kolom tanggal (mencari nama kolom 'TGL' atau 'Tanggal')
    kolom_tgl = [col for col in df.columns if 'TGL' in str(col).upper() or 'TANGGAL' in str(col).upper()]
    
    if not kolom_tgl:
        st.error("Kolom tanggal tidak ditemukan di dalam sheet Excel.")
    else:
        nama_kolom_tgl = kolom_tgl[0]
        data_hari = df[df[nama_kolom_tgl] == tanggal_pilihan]

        if data_hari.empty:
            st.warning(f"Data untuk tanggal {tanggal_pilihan} {bulan_pilihan} tidak ditemukan.")
        else:
            # Ambil kolom jam (0..23 atau 00..23)
            kolom_jam = [col for col in df.columns if str(col).isdigit() or (isinstance(col, int) and 0 <= col <= 23)]
            
            jam_list = [f"{str(j).zfill(2)}:00" for j in kolom_jam]
            tinggi_list = [float(data_hari[j].values[0]) for j in kolom_jam]

            # Membuat DataFrame untuk grafik dan tabel
            df_plot = pd.DataFrame({
                "Jam": jam_list,
                "Tinggi (m)": tinggi_list
            })

            # Menghitung Pasang Maksimum & Surut Minimum
            val_max = df_plot["Tinggi (m)"].max()
            jam_max = df_plot.loc[df_plot["Tinggi (m)"] == val_max, "Jam"].values[0]
            
            val_min = df_plot["Tinggi (m)"].min()
            jam_min = df_plot.loc[df_plot["Tinggi (m)"] == val_min, "Jam"].values[0]

            # ==========================================
            # 6. TAMPILAN METRIK RINGKASAN
            # ==========================================
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("📅 Tanggal Pilih", f"{tanggal_pilihan} {bulan_pilihan} 2026")
            col2.metric("🔺 Pasang Maksimum", f"{val_max:.2f} m", f"Pukul {jam_max} WIB")
            col3.metric("🔻 Surut Minimum", f"{val_min:.2f} m", f"Pukul {jam_min} WIB")
            st.markdown("---")

            # ==========================================
            # 7. GRAFIK LINE CHART (PLOTLY)
            # ==========================================
            fig = go.Figure()

            # Plot garis utama pasang surut
            fig.add_trace(go.Scatter(
                x=df_plot["Jam"],
                y=df_plot["Tinggi (m)"],
                mode='lines+markers',
                name='Ketinggian Air (m)',
                line=dict(color='#0066cc', width=3),
                marker=dict(size=6)
            ))

            # Penanda Titik Pasang Maksimum
            fig.add_trace(go.Scatter(
                x=[jam_max], y=[val_max],
                mode='markers+text',
                name='Pasang Maksimum',
                text=[f"Max: {val_max:.2f}m"],
                textposition="top center",
                marker=dict(color='red', size=12, symbol='triangle-up')
            ))

            # Penanda Titik Surut Minimum
            fig.add_trace(go.Scatter(
                x=[jam_min], y=[val_min],
                mode='markers+text',
                name='Surut Minimum',
                text=[f"Min: {val_min:.2f}m"],
                textposition="bottom center",
                marker=dict(color='green', size=12, symbol='triangle-down')
            ))

            fig.update_layout(
                title=f"Grafik Pasang Surut Air Laut - {tanggal_pilihan} {bulan_pilihan} 2026",
                xaxis_title="Waktu (WIB)",
                yaxis_title="Ketinggian Air (Meter)",
                hovermode="x unified",
                template="plotly_white",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # ==========================================
            # 8. TABEL DATA DETAIL
            # ==========================================
            with st.expander("📊 Lihat Tabel Data Per Jam"):
                st.dataframe(df_plot.set_index("Jam").T, use_container_width=True)
