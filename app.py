import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. MEMBUAT DATASET PASANG SURUT 2026
# ==========================================
print("1. Menggenerate data pasang surut 2026...")

# Rentang waktu sepanjang tahun 2026 (per jam)
time_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-12-31 23:00:00', freq='h')

# Simulasi komponen pasang surut (Komponen Utama M2, S2, K1, O1)
# Menggunakan formula gelombang sinusoidal pasang surut diurnal/semidiurnal
hours = np.arange(len(time_range))
tide_m2 = 0.8 * np.cos(2 * np.pi * hours / 12.42)          # Semidiurnal utama
tide_s2 = 0.3 * np.cos(2 * np.pi * hours / 12.00)          # Semidiurnal matahari
tide_k1 = 0.9 * np.cos(2 * np.pi * hours / 23.93 + 0.5)    # Diurnal utama
tide_o1 = 0.5 * np.cos(2 * np.pi * hours / 25.82 - 0.3)    # Diurnal bulan

# Mean Sea Level (MSL) rata-rata Probolinggo (~1.4 m)
msl = 1.4

# Menghitung Sea Water Level dasar
sea_level_simulated = msl + tide_m2 + tide_s2 + tide_k1 + tide_o1

# Menambahkan noise acak kecil (variasi cuaca/angin)
np.random.seed(42)
noise = np.random.normal(0, 0.05, len(time_range))
sea_level_simulated += noise

# Membuat DataFrame
df = pd.DataFrame({
    'Timestamp': time_range,
    'Year': time_range.year,
    'Month': time_range.month,
    'Day': time_range.day,
    'Hour': time_range.hour,
    'DayOfWeek': time_range.dayofweek,
    'DayOfYear': time_range.dayofyear,
    'Sea_Level_m': np.round(sea_level_simulated, 2)
})

# ==========================================
# 2. PELATIHAN MODEL MACHINE LEARNING
# ==========================================
print("2. Melatih model Machine Learning (Random Forest)...")

# Memisahkan Fitur (X) dan Target (y)
X = df[['Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear']]
y = df['Sea_Level_m']

# Split data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan pelatihan model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi pada data uji untuk evaluasi
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"   - Evaluation RMSE : {rmse:.4f} m")
print(f"   - Evaluation R2   : {r2:.4f}")

# Memasukkan hasil prediksi model ke dalam DataFrame utama
df['ML_Predicted_Sea_Level_m'] = np.round(model.predict(X), 2)

# ==========================================
# 3. EKSPOR KE CSV & EXCEL
# ==========================================
print("3. Mengekspor data ke format CSV dan Excel...")

# Ekspor ke CSV
csv_filename = 'sea_water_level_probolinggo_2026.csv'
df.to_csv(csv_filename, index=False)
print(f"   - File CSV berhasil dibuat: {csv_filename}")

# Ekspor ke Excel
excel_filename = 'sea_water_level_probolinggo_2026.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Full_Data_2026', index=False)
    
    # Membuat sheet ringkasan bulanan (Min, Max, Rata-rata)
    monthly_summary = df.groupby('Month')['Sea_Level_m'].agg(
        Max_High_Tide_m='max',
        Min_Low_Tide_m='min',
        Average_Level_m='mean'
    ).reset_index()
    monthly_summary.to_excel(writer, sheet_name='Monthly_Summary', index=False)

print(f"   - File Excel berhasil dibuat: {excel_filename}")

# ==========================================
# 4. VISUALISASI GRAFIK
# ==========================================
print("4. Menampilkan grafik pasang surut...")

plt.figure(figsize=(15, 6))

# Plot sampel data bulan Januari 2026 agar grafik terlihat detail
df_january = df[df['Month'] == 1]

plt.plot(df_january['Timestamp'], df_january['Sea_Level_m'], label='Simulated Sea Level', color='navy', alpha=0.7, linewidth=1.5)
plt.plot(df_january['Timestamp'], df_january['ML_Predicted_Sea_Level_m'], label='ML Predicted Sea Level', color='red', linestyle='--', alpha=0.8, linewidth=1)

plt.title('Prediksi Sea Water Level Probolinggo - Bulan Januari 2026', fontsize=14, fontweight='bold')
plt.xlabel('Tanggal / Waktu', fontsize=12)
plt.ylabel('Sea Level (Meter)', fontsize=12)
plt.axhline(y=1.4, color='green', linestyle=':', label='Mean Sea Level (1.4m)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()

# Tampilkan grafik
plt.show()

print("Proses selesai!")
