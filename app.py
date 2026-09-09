import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Machine Learning Pasang Surut Air Laut",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Prediksi Pasang Surut ML vs Data Realtime/Observasi")
st.markdown("Lokasi: **Selat Madura / Probolinggo** (`S 07°38.659' E 113°01.641'`)")

# ==========================================
# 2. SIDEBAR - FILE UPLOADER & FILTER
# ==========================================
st.sidebar.header("⚙️ Pengaturan & Filter")

uploaded_file = st.sidebar.file_uploader("Upload File Excel Pasang Surut", type=["xlsx", "xls"])
FILE_EXCEL = uploaded_file if uploaded_file is not None else "Tabel_Pasang_Surut_Probolinggo_2026.xlsx"

# Mapping Bulan
MAP_BULAN = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, 
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8, 
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12
}

# ==========================================
# 3. FUNGSI PARSING EXCEL YANG FLEKSIBEL
# ==========================================
@st.cache_data
def build_ml_dataset(file_source):
    """Membaca semua sheet Excel secara fleksibel tanpa bergantung nama sheet persis."""
    dataset = []
    
    try:
        excel_file = pd.ExcelFile(file_source)
        sheet_names = excel_file.sheet_names
        
        for idx_sheet, sheet_name in enumerate(sheet_names):
            # Tentukan angka bulan (dari nama sheet atau urutan sheet)
            angka_bulan = None
            for nama_b, angka_b in MAP_BULAN.items():
                if nama_b.lower() in str(sheet_name).lower():
                    angka_bulan = angka_b
                    break
            
            if angka_bulan is None:
                # Jika nama sheet berupa angka atau urutan (1..12)
                if str(sheet_name).strip().isdigit() and 1 <= int(sheet_name) <= 12:
                    angka_bulan = int(sheet_name)
                elif idx_sheet < 12:
                    angka_bulan = idx_sheet + 1
                else:
                    continue

            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Cari baris header yang berisi kolom Tanggal/TGL
            kolom_tgl = [col for col in df.columns if any(k in str(col).upper() for k in ['TGL', 'TANGGAL', 'DATE', 'DAY'])]
            
            if not kolom_tgl:
                continue
                
            nama_col_tgl = kolom_tgl[0]
            
            # Cari kolom jam (0..23 atau berformat '00:00')
            kolom_jam_map = {}
            for col in df.columns:
                col_str = str(col).strip()
                # Ekstrak angka jam jika ada
                if col_str.isdigit() and 0 <= int(col_str) <= 23:
                    kolom_jam_map[col] = int(col_str)
                elif ':' in col_str:
                    jam_part = col_str.split(':')[0]
                    if jam_part.isdigit() and 0 <= int(jam_part) <= 23:
                        kolom_jam_map[col] = int(jam_part)

            # Extract data jam-jaman
            for _, row in df.iterrows():
                tgl_val = row[nama_col_tgl]
                if pd.notna(tgl_val) and str(tgl_val).strip().isdigit():
                    tgl_num = int(tgl_val)
                    if 1 <= tgl_num <= 31:
                        for col_orig, jam_num in kolom_jam_map.items():
                            val_tinggi = row[col_orig]
                            if pd.notna(val_tinggi):
                                try:
                                    dataset.append({
                                        "Bulan": int(angka_bulan),
                                        "Tanggal": int(tgl_num),
                                        "Jam": int(jam_num),
                                        "Tinggi_Air": float(val_tinggi)
                                    })
                                except ValueError:
                                    continue
    except Exception as e:
        st.error(f"Error membaca file Excel: {e}")
        return pd.DataFrame()

    return pd.DataFrame(dataset)

# ==========================================
# 4. TRAINING & PREDIKSI
# ==========================================
df_dataset = build_ml_dataset(FILE_EXCEL)

if df_dataset.empty:
    st.error("⚠️ Dataset kosong atau format file Excel tidak dapat terbaca.")
    st.info("💡 **Solusi:** Silakan unggah (*upload*) file Excel `.xlsx` Anda menggunakan menu di **Sidebar sebelah kiri**.")
else:
    # Train Model Random Forest
    X = df_dataset[["Bulan", "Tanggal", "Jam"]]
    y = df_dataset["Tinggi_Air"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_ml = RandomForestRegressor(n_estimators=100, random_state=42)
    model_ml.fit(X_train, y_train)
    
    y_pred_test = model_ml.predict(X_test)
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_score_val = r2_score(y_test, y_pred_test)

    # Filter Sidebar
    bulan_nama = st.sidebar.selectbox("Pilih Bulan", list(MAP_BULAN.keys()), index=8)
    bulan_angka = MAP_BULAN[bulan_nama]
    tanggal_input = st.sidebar.number_input("Pilih Tanggal", min_value=1, max_value=31, value=9)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Performa Model ML")
    st.sidebar.write(f"**Akurasi (R² Score):** `{r2_score_val * 100:.2f}%`")
    st.sidebar.write(f"**RMSE Error:** `{rmse_score:.3f} m`")

    # Data Aktual
    df_aktual = df_dataset[(df_dataset["Bulan"] == bulan_angka) & (df_dataset["Tanggal"] == tanggal_input)].sort_values("Jam")

    if df_aktual.empty:
        st.warning(f"Data aktual untuk tanggal {tanggal_input} {bulan_nama} tidak ditemukan dalam Excel.")
    else:
        jam_range = list(range(24))
        X_predict = pd.DataFrame({
            "Bulan": [bulan_angka] * 24,
            "Tanggal": [tanggal_input] * 24,
            "Jam": jam_range
        })

        y_pred_ml = model_ml.predict(X_predict)

        df_hasil = pd.DataFrame({
            "Jam": [f"{str(j).zfill(2)}:00" for j in jam_range],
            "Data Realtime/Observasi (m)": df_aktual["Tinggi_Air"].values if len(df_aktual) == 24 else y_pred_ml,
            "Prediksi ML (m)": y_pred_ml,
            "Selisih Error (m)": np.abs(df_aktual["Tinggi_Air"].values - y_pred_ml) if len(df_aktual) == 24 else 0
        })

        # Display Metrik
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tanggal Selected", f"{tanggal_input} {bulan_nama} 2026")
        col2.metric("Max Realtime", f"{df_hasil['Data Realtime/Observasi (m)'].max():.2f} m")
        col3.metric("Max Prediksi ML", f"{df_hasil['Prediksi ML (m)'].max():.2f} m")
        col4.metric("Rata-rata Error", f"{df_hasil['Selisih Error (m)'].mean():.3f} m")

        st.markdown("---")

        # Plotly Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hasil["Jam"], y=df_hasil["Data Realtime/Observasi (m)"],
            mode='lines+markers', name='Data Realtime / Observasi',
            line=dict(color='#0066cc', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=df_hasil["Jam"], y=df_hasil["Prediksi ML (m)"],
            mode='lines+markers', name='Prediksi Machine Learning',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        fig.update_layout(
            title=f"Perbandingan ML vs Realtime Pasang Surut ({tanggal_input} {bulan_nama} 2026)",
            xaxis_title="Waktu (WIB)", yaxis_title="Ketinggian Air (Meter)",
            hovermode="x unified", template="plotly_white", height=480
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Lihat Tabel Komparasi Hasil Prediksi vs Realtime"):
            st.dataframe(df_hasil.set_index("Jam"), use_container_width=True)
