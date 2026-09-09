import pandas as pd

def baca_pasang_surut(file_path, bulan, tanggal):
    """
    Fungsi untuk mengambil data pasang surut berdasarkan bulan dan tanggal.
    
    Parameters:
        file_path (str): Jalur/nama file Excel
        bulan (str): Nama sheet bulan (contoh: 'September')
        tanggal (int): Tanggal yang dicari (1 - 31)
    """
    try:
        # Membaca sheet sesuai bulan yang dipilih
        df = pd.read_excel(file_path, sheet_name=bulan)
        
        # Mencari baris data berdasarkan tanggal (kolom 'TGL' atau 'Tanggal')
        # Menyesuaikan nama kolom tanggal di sheet
        kolom_tgl = [col for col in df.columns if 'TGL' in str(col).upper() or 'TANGGAL' in str(col).upper()][0]
        data_hari = df[df[kolom_tgl] == tanggal]
        
        if data_hari.empty:
            print(f"Data untuk tanggal {tanggal} {bulan} tidak ditemukan.")
            return
        
        print(f"=== PREDIKSI PASANG SURUT PROBOLINGGO ===")
        print(f"Tanggal: {tanggal} {bulan} 2026\n")
        
        # Mengambil kolom jam (biasanya berformat 00, 01, ..., 23)
        kolom_jam = [col for col in df.columns if str(col).isdigit() or (isinstance(col, int) and 0 <= col <= 23)]
        
        print("Jam (WIB) | Ketinggian (m)")
        print("-" * 27)
        
        tinggi_air = []
        for jam in kolom_jam:
            val = data_hari[jam].values[0]
            tinggi_air.append((jam, val))
            print(f"  {str(jam).zfill(2)}:00    |    {val:.2f} m")
            
        # Mencari pasang tertinggi dan surut terendah
        jam_max, val_max = max(tinggi_air, key=lambda x: x[1])
        jam_min, val_min = min(tinggi_air, key=lambda x: x[1])
        
        print("-" * 27)
        print(f"Pasang Maksimum : {val_max:.2f} m (Pukul {str(jam_max).zfill(2)}:00 WIB)")
        print(f"Surut Minimum   : {val_min:.2f} m (Pukul {str(jam_min).zfill(2)}:00 WIB)")

    except Exception as e:
        print(f"Error membaca file: {e}")

# --- CONTOH PENGGUNAAN ---
# Ganti 'Tabel_Pasang_Surut_Probolinggo_2026.xlsx' sesuai nama file Anda
file_excel = "Tabel_Pasang_Surut_Probolinggo_2026.xlsx"

# Memanggil data tanggal 9 September 2026
baca_pasang_surut(file_excel, bulan="September", tanggal=9)
