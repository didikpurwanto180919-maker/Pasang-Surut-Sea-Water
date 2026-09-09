import streamlit as st
import pandas as pd

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Pasang Surut Air Laut Probolinggo",
    page_icon="🌊",
    layout="wide"
)

# 2. FUNGSI MEMBACA DATA
@st.cache_data
def load_data(file_path, sheet_name):
    try:
        # Menggunakan engine openpyxl atau pandas default
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"Gagal membaca sheet '{sheet_name}': {e}")
        return None

# 3. HEADER & TAMPILAN UTAMA
st.title("🌊 Aplikasi Prediksi Pasang Surut Air Laut")
st.markdown("Stasiun **Probolinggo** | Lintang `07° 44' 10.79\" S` | Bujur `113° 12' 59.64\" T` (GMT +07:00)")

# 4. SIDEBAR (FILTER INPUT)
st.sidebar.header("⚙️ Filter Data")

FILE_EXCEL = "Tabel_Pasang_Surut_Probolinggo_2026.xlsx"
DAFTAR_BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", 
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

bulan_pilihan = st.sidebar.selectbox("Pilih Bulan", DAFTAR_BULAN, index=8)
tanggal_pilihan = st.sidebar.number_input("Pilih Tanggal", min_value=1, max_value=31, value=9)

# 5. PEMROSESAN DATA
df = load_data(FILE_EXCEL, bulan_pilihan)

if df is not None:
    kolom_tgl = [col for col in df.columns if 'TGL' in str(col).upper() or 'TANGGAL' in str(col).upper()]
    
    if not kolom_tgl:
        st.error("Kolom tanggal tidak ditemukan.")
    else:
        nama_kolom_tgl = kolom_tgl[0]
        data_hari = df[df[nama_kolom_tgl] == tanggal_pilihan]

        if data_hari.empty:
            st.warning(f"Data untuk tanggal {tanggal_pilihan} {bulan_pilihan} tidak ditemukan.")
        else:
            kolom_jam = [col for col in df.columns if str(col).isdigit() or (isinstance(col, int) and 0 <= col <= 23)]
            
            jam_list = [f"{str(j).zfill(2)}:00" for j in kolom_jam]
            tinggi_list = [float(data_hari[j].values[0]) for j in kolom_jam]

            df_plot = pd.DataFrame({
                "Jam": jam_list,
                "Tinggi Air (m)": tinggi_list
            }).set_index("Jam")

            # Ringkasan Ekstrem
            val_max = df_plot["Tinggi Air (m)"].max()
            jam_max = df_plot[df_plot["Tinggi Air (m)"] == val_max].index[0]
            
            val_min = df_plot["Tinggi Air (m)"].min()
            jam_min = df_plot[df_plot["Tinggi Air (m)"] == val_min].index[0]

            # Display Metrik
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("📅 Tanggal Pilih", f"{tanggal_pilihan} {bulan_pilihan} 2026")
            col2.metric("🔺 Pasang Maksimum", f"{val_max:.2f} m", f"Pukul {jam_max} WIB")
            col3.metric("🔻 Surut Minimum", f"{val_min:.2f} m", f"Pukul {jam_min} WIB")
            st.markdown("---")

            # 6. GRAFIK LINE CHART NATIVE STREAMLIT
            st.subheader(f"Grafik Pasang Surut - {tanggal_pilihan} {bulan_pilihan} 2026")
            st.line_chart(df_plot["Tinggi Air (m)"])

            # 7. TABEL DATA DETAIL
            with st.expander("📊 Lihat Tabel Data Per Jam"):
                st.dataframe(df_plot.T, use_container_width=True)
