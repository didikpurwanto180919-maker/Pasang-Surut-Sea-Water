import pandas as pd
import matplotlib.pyplot as plt

# 1. Menyiapkan Data Ringkasan Ekstrem Bulanan (Sep - Des 2026)
data_ekstrem = {
    "Bulan": ["September 2026", "Oktober 2026", "November 2026", "Desember 2026"],
    "Koreksi Elevasi (cm)": [-8, -1, 10, 16],
    "Pasang Tertinggi (m)": [2.8, 2.9, 3.0, 3.0],
    "Jam Pasang Peak": ["22:00 - 23:00", "22:00 - 23:00", "22:00 - 23:00", "22:00 - 23:00"],
    "Surut Terendah (m)": [0.2, 0.0, -0.1, -0.1],
    "Jam Surut Low": ["05:00 - 06:00", "05:00 - 06:00", "04:00 - 06:00", "04:00 - 06:00"],
    "Hari Maksimum": ["2–5 & 29–30", "1–4 & 28–31", "2–5 & 27–30", "3–6 & 28–31"]
}

df_ekstrem = pd.DataFrame(data_ekstrem)

print("=== TABEL RINGKASAN EKSTREM PASANG SURUT PROBOLINGGO (SEP - DES 2026) ===")
print(df_ekstrem.to_string(index=False))
print("\n" + "="*70 + "\n")

# 2. Simulasi/Pemodelan Gelombang Pasang Surut Harian (Siklus Diurnal)
# Membuat kurva sampel 24 jam untuk menggambarkan pola pasang malam & surut subuh
jam = list(range(24))

# Aproksimasi ketinggian air harian (sampel November/Desember 2026)
# Puncak (~3.0m) jam 22:00, Lembah (~-0.1m) jam 05:00
import math
ketinggian_air = []
for h in jam:
    # Menggunakan fungsi sinus dipatenkan untuk siklus harian 24 jam
    val = 1.45 + 1.55 * math.sin((h - 13.5) * (2 * math.pi / 24))
    ketinggian_air.append(round(val, 2))

# 3. Visualisasi Grafik Siklus Harian
plt.figure(figsize=(10, 5))
plt.plot(jam, ketinggian_air, marker='o', color='b', linestyle='-', linewidth=2, label='Ketinggian Air (m)')
plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='Chart Datum / LWS (0.0 m)')

# Format Grafik
plt.title("Simulasi Pola Harian Pasang Surut Probolinggo (Diurnal Tide)", fontsize=12, fontweight='bold')
plt.xlabel("Jam (WIB)", fontsize=10)
plt.ylabel("Ketinggian Air di atas LWS (meter)", fontsize=10)
plt.xticks(range(0, 24, 2))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')

# Tanda Puncak & Lembah
plt.annotate('Pasang Peak (~3.0 m)\n(21:00 - 23:00)', xy=(22, 3.0), xytext=(17, 2.7),
             arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6))
plt.annotate('Surut Low (~-0.1 m)\n(04:00 - 06:00)', xy=(5, -0.1), xytext=(7, 0.3),
             arrowprops=dict(facecolor='orange', shrink=0.05, width=1, headwidth=6))

plt.tight_layout()
plt.show()
