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
# 2. LOAD & PREPARE DATASET UNTUK ML
# ==========================================
FILE_EXCEL = "Tabel_Pasang_Surut_Probolinggo_2026.xlsx"

DAFTAR_BULAN = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, 
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8, 
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12
}

@st.cache_data
def build_ml_dataset(file_path):
    """Mengubah format tabel Excel bulanan menjadi dataset ML (Bulan, Tanggal, Jam -> Tinggi Air)."""
    dataset = []
    
    for nama_bulan, angka_bulan in DAFTAR_BULAN.items():
        try:
            df = pd.read_excel(file_path, sheet_name=nama_bulan)
            kolom_tgl = [col for col in df.columns if 'TGL' in str(col).upper() or 'TANGGAL' in str(col).upper()][0]
            kolom_jam = [col for col in df.columns if str(col).isdigit() or (isinstance(col, int) and 0 <= col <= 23)]
            
            for _, row in df.iterrows():
                tgl = row[kolom_tgl]
                if pd.notna(tgl):
                    for jam in kolom_jam:
                        tinggi = row[jam]
                        if pd.notna(tinggi):
                            dataset.append({
                                "Bulan": int(angka_bulan),
                                "Tanggal": int(tgl),
                                "Jam": int(jam),
                                "Tinggi_Air": float(tinggi)
                            })
        except Exception:
            continue
            
    return pd.DataFrame(dataset)

# ==========================================
# 3. TRAINING MODEL MACHINE LEARNING
# ==========================================
@st.cache_resource
def train_ml_model(df_all):
    """Melatih model Random Forest Regressor."""
    X = df_all[["Bulan", "Tanggal", "Jam"]]
    y = df_all["Tinggi_Air"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2

# Try-Except membaca dataset
try:
    df_dataset = build_ml_dataset(FILE_EXCEL)
    
    if df_dataset.empty:
        st.error("Dataset kosong atau file Excel tidak sesuai format.")
        st.stop()
        
    model_ml, rmse_score, r2_score_val = train_ml_model(df_dataset)

    # ==========================================
    # 4. SIDEBAR FILTER & EVALUASI
    # ==========================================
    st.sidebar.header("⚙️ Filter Prediksi ML")
    
    bulan_nama = st.sidebar.selectbox("Pilih Bulan", list(DAFTAR_BULAN.keys()), index=8) # Default September
    bulan_angka = DAFTAR_BULAN[bulan_nama]
    tanggal_input = st.sidebar.number_input("Pilih Tanggal", min_value=1, max_value=31, value=9)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Performa Model ML")
    st.sidebar.write(f"**Akurasi (R² Score):** `{r2_score_val * 100:.2f}%`")
    st.sidebar.write(f"**RMSE (Error):** `{rmse_score:.3f} meter`")

    # ==========================================
    # 5. PREDIKSI VS REALTIME / OBSERVASI
    # ==========================================
    # Filter data aktual (realtime/observasi dari Excel)
    df_aktual = df_dataset[(df_dataset["Bulan"] == bulan_angka) & (df_dataset["Tanggal"] == tanggal_input)].sort_values("Jam")

    if df_aktual.empty:
        st.warning(f"Data aktual/observasi untuk tanggal {tanggal_input} {bulan_nama} tidak ditemukan.")
    else:
        # Menyiapkan input jam (0..23) untuk prediksi ML
        jam_range = list(range(24))
        X_predict = pd.DataFrame({
            "Bulan": [bulan_angka] * 24,
            "Tanggal": [tanggal_input] * 24,
            "Jam": jam_range
        })

        # Jalankan Prediksi ML
        y_pred_ml = model_ml.predict(X_predict)

        # Buat DataFrame Gabungan
        df_hasil = pd.DataFrame({
            "Jam": [f"{str(j).zfill(2)}:00" for j in jam_range],
            "Data Realtime/Observasi (m)": df_aktual["Tinggi_Air"].values,
            "Prediksi ML (m)": y_pred_ml,
            "Selisih / Error (m)": np.abs(df_aktual["Tinggi_Air"].values - y_pred_ml)
        })

        # ==========================================
        # 6. METRIK KINERJA PREDIKSI HARI INI
        # ==========================================
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tanggal Input", f"{tanggal_input} {bulan_nama} 2026")
        col2.metric("Max Realtime", f"{df_hasil['Data Realtime/Observasi (m)'].max():.2f} m")
        col3.metric("Max Prediksi ML", f"{df_hasil['Prediksi ML (m)'].max():.2f} m")
        col4.metric("Rata-rata Selisih Error", f"{df_hasil['Selisih / Error (m)'].mean():.3f} m")

        st.markdown("---")

        # ==========================================
        # 7. GRAFIK PLOTLY PREDIKSI VS REALTIME
        # ==========================================
        fig = go.Figure()

        # Garis Data Realtime / Observasi
        fig.add_trace(go.Scatter(
            x=df_hasil["Jam"],
            y=df_hasil["Data Realtime/Observasi (m)"],
            mode='lines+markers',
            name='Data Realtime / Observasi',
            line=dict(color='#0066cc', width=3),
            marker=dict(size=6)
        ))

        # Garis Prediksi Machine Learning
        fig.add_trace(go.Scatter(
            x=df_hasil["Jam"],
            y=df_hasil["Prediksi ML (m)"],
            mode='lines+markers',
            name='Prediksi Machine Learning',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6, symbol='x')
        ))

        fig.update_layout(
            title=f"Perbandingan Prediksi ML vs Realtime Pasang Surut ({tanggal_input} {bulan_nama} 2026)",
            xaxis_title="Waktu (WIB)",
            yaxis_title="Ketinggian Air (Meter)",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # 8. TABEL DETAIL DENGAN ERROR
        # ==========================================
        with st.expander("📋 Lihat Tabel Komparasi Hasil Prediksi vs Realtime"):
            st.dataframe(df_hasil.set_index("Jam"), use_container_width=True)

except FileNotFoundError:
    st.error("File Excel `Tabel_Pasang_Surut_Probolinggo_2026.xlsx` tidak ditemukan. Pastikan file tersimpan di direktori aplikasi.")
