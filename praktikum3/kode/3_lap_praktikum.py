import pandas as pd
df = pd.read_csv("data_penjualan.csv")
print(df.head())
# Hapus kolom yang tidak diperlukan
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Hapus data duplikat 
df = df.drop_duplicates()

# Rename kolom 
df = df.rename(columns={"qty": "jumlah", "price": "harga"})

print("--- Data setelah rename & drop duplicates ---")
print(df.head())

#Catat berapa banyak nilai kosong SEBELUM
print("--- Nilai kosong SEBELUM perbaikan ---")
print(df.isnull().sum())

#Hitung rata-rata kolom umur 
rata_rata_umur = df["umur"].mean()
print(f"\nRata-rata umur: {rata_rata_umur}")

#Isi nilai kosong di kolom umur dengan rata-rata 
df["umur"] = df["umur"].fillna(rata_rata_umur)

#Isi nilai kosong di kolom jumlah dengan 0
df["jumlah"] = df["jumlah"].fillna(0)

# Catat berapa banyak nilai kosong SESUDAH
print("\n--- Nilai kosong SESUDAH perbaikan ---")
print(df.isnull().sum())
# Replace nilai kategorikal 
df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})

# Normalisasi kolom harga 
df["harga_norm"] = (df["harga"] - df["harga"].min()) / \
                   (df["harga"].max() - df["harga"].min())

# Feature engineering (total_price) 
df["total_price"] = df["jumlah"] * df["harga"]

print("--- Data setelah Transformasi ---")
print(df.head())

# Latihan 1: interpolate()
print("\n--- Latihan 1: interpolate() ---")
# Kita baca ulang data mentah untuk demo
df_latihan_1 = pd.read_csv("data_penjualan.csv")

print("Data 'umur' SEBELUM interpolate (baris 0-5):")
print(df_latihan_1[['nama', 'umur']].head())

# Gunakan interpolate()
df_latihan_1['umur'] = df_latihan_1['umur'].interpolate()

print("\nData 'umur' SESUDAH interpolate (baris 0-5):")
print(df_latihan_1[['nama', 'umur']].head())

# Latihan 2: describe() Sebelum vs Sesudah
print("\n--- Latihan 2: describe() Sebelum vs Sesudah ---")
# Ringkasan statistik SEBELUM cleaning
df_awal = pd.read_csv("data_penjualan.csv")

print("\n--- Ringkasan Statistik SEBELUM Cleaning (df_awal) ---")
print(df_awal.describe())

# Ringkasan statistik SESUDAH cleaning
print("\n--- Ringkasan Statistik SESUDAH Cleaning (df) ---")
print(df.describe())

print("\nPerbedaan utama:")
print("- 'count' (jumlah data) di df (sesudah) lebih tinggi untuk 'umur' dan 'jumlah' karena NaN sudah diisi.")
print("- 'count' total baris di df (159) lebih rendah dari df_awal (163) karena 4 baris duplikat dihapus.")
print("- 'mean', 'std', 'min' 'umur' dan 'jumlah' (qty) berubah karena nilai NaN sudah diisi (diisi mean & 0).")
print("- 'df' (sesudah) memiliki kolom baru: 'harga_norm' dan 'total_price'.")