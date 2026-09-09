import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 1. PARAMETER KONSTANTA HARMONIK (Dari Tabel Pasang Surut Probolinggo)
# So = Mean Sea Level (MSL) dalam meter (410 cm = 4.10 m)
S0 = 4.10 

# Data Amplitudo (A) dalam meter dan Beda Fase (g) dalam derajat
# Amplitudo diubah dari cm ke meter (dibagi 100)
harmonics = {
    'M2':  {'A': 0.20, 'g': 247.0, 'speed': 28.9841042}, # Kecepatan sudut (deg/jam)
    'S2':  {'A': 0.16, 'g': 288.0, 'speed': 30.0000000},
    'N2':  {'A': 0.05, 'g': 231.0, 'speed': 28.4397295},
    'K2':  {'A': 0.04, 'g': 288.0, 'speed': 30.0821373},
    'K1':  {'A': 0.53, 'g': 250.0, 'speed': 15.0410686},
    'O1':  {'A': 0.38, 'g': 187.0, 'speed': 13.9430356},
    'P1':  {'A': 0.18, 'g': 250.0, 'speed': 14.9589314},
    'M4':  {'A': 0.01, 'g': 180.0, 'speed': 57.9682084},
    'MS4': {'A': 0.01, 'g': 220.0, 'speed': 58.9841042}
}

def hitung_tinggi_air(jam_ke_t):
    """
    Menghitung tinggi muka air (m) pada jam ke-t dari titik acuan awal
    """
    h = S0
    for constituent, data in harmonics.items():
        A = data['A']
        g = np.radians(data['g']) # Konversi fase ke radian
        speed = np.radians(data['speed']) # Konversi kecepatan sudut ke radian/jam
        
        # Komponen gelombang pasut
        h += A * np.cos(speed * jam_ke_t - g)
    return h

# 2. SIMULASI PREDIKSI
# Tentukan rentang waktu simulasi
start_date = datetime(2026, 9, 9, 0, 0) # Waktu mulai (YYYY, MM, DD, HH, MM)
hours_to_predict = 48                    # Durasi prediksi (48 jam / 2 hari)

data_prediksi = []

for hour in range(hours_to_predict):
    current_time = start_date + timedelta(hours=hour)
    height = hitung_tinggi_air(hour)
    data_prediksi.append({
        'Waktu (WIB)': current_time.strftime('%Y-%m-%d %H:%M'),
        'Tinggi Air (m)': round(height, 2)
    })

# 3. TAMPILKAN HASIL PREDIKSI
df = pd.DataFrame(data_prediksi)
print("--- PREDIKSI PASANG SURUT PROBOLINGGO ---")
print(df.head(10)) # Menampilkan 10 jam pertama

# 4. MEMASUKKAN GRAFIK (OPTIONAL)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    plt.plot(df['Waktu (WIB)'], df['Tinggi Air (m)'], color='blue', marker='o', markersize=3)
    plt.axhline(y=S0, color='red', linestyle='--', label=f'MSL / S0 ({S0} m)')
    
    plt.xticks(rotation=45)
    plt.title('Prediksi Elevasi Pasang Surut Stasiun Probolinggo')
    plt.xlabel('Waktu')
    plt.ylabel('Tinggi Air (m)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
except ImportError:
    print("\nInstall 'matplotlib' jika ingin menampilkan grafik visual.")
