import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. KONFIGURASI METADATA & LOKASI
# ==========================================
STATION_NAME = "Pasuruan"
LATITUDE = -7.644317   # S 07° 38.659'
LONGITUDE = 113.02735  # E 113° 01.641'
API_URL = "https://srgi.big.go.id/api/pasut/pasuruan" # Sesuaikan Endpoint API/Token BIG Anda

# ==========================================
# 2. SIMULASI / INGESTION DATA REAL-TIME
# ==========================================
def fetch_realtime_pasut_data(api_url):
    """
    Mengambil data dari API SRGI/BIG. 
    Menggunakan fallback data sintetis harmonik jika API butuh autentikasi.
    """
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df[['water_level']]
    except Exception as e:
        print(f"[Info] API BIG tidak dapat diakses langsung tanpa Token, menggenerasi simulasi data pasut Pasuruan: {e}")

    # Simulasi Data Pasang Surut (Komponen Utama M2 & S2 khas Selat Madura / Pasuruan)
    time_range = pd.date_range(start="2026-08-01", periods=1000, freq="15min")
    t = np.arange(len(time_range))
    
    # Komponen Pasut Semidiurnal + Noise Cuaca BMKG
    m2_tide = 1.2 * np.sin(2 * np.pi * t / 49.6)   # Siklus utama ~12.4 jam
    s2_tide = 0.5 * np.sin(2 * np.pi * t / 48.0)   # Siklus kedua ~12.0 jam
    weather_noise = np.random.normal(0, 0.08, len(t)) # Distorisi dinamika pesisir
    
    water_level = 2.0 + m2_tide + s2_tide + weather_noise # Mean Sea Level ~2m
    
    df = pd.DataFrame({'water_level': water_level}, index=time_range)
    return df

# Download data
df = fetch_realtime_pasut_data(API_URL)

# ==========================================
# 3. PREPROCESSING DATA
# ==========================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['water_level']])

# Mengubah data ke format Window (Misal: 24 step/6 jam terakhir untuk prediksi 1 step ke depan)
LOOKBACK = 24

def create_dataset(dataset, lookback=1):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:(i + lookback), 0])
        Y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_data, LOOKBACK)

# Reshape untuk LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split Train (80%) & Test (20%)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ==========================================
# 4. PEMBUATAN & PELATIHAN MODEL LSTM
# ==========================================
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("--- Melatih Model Machine Learning ---")
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

# ==========================================
# 5. PREDIKSI & EVALUASI
# ==========================================
predictions = model.predict(X_test)

# Denormalisasi kembali ke satuan Meter
predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# ==========================================
# 6. VISUALISASI PREDIKSI VS AKTUAL
# ==========================================
test_timestamps = df.index[train_size + LOOKBACK:]

plt.figure(figsize=(14, 6))
plt.plot(test_timestamps, y_test_actual, label="Aktual Level Air Laut (BIG)", color="blue", linewidth=1.5)
plt.plot(test_timestamps, predictions_actual, label="Prediksi Machine Learning", color="red", linestyle="--", linewidth=1.5)

plt.title(f"Prediksi vs Aktual Level Air Laut - Stasiun {STATION_NAME} (Lat: {LATITUDE}, Lon: {LONGITUDE})")
plt.xlabel("Waktu")
plt.ylabel("Tinggi Muka Air (Meter)")
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
