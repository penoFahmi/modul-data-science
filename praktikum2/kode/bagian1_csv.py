# Import semua library di bagian atas file
import pandas as pd
import matplotlib.pyplot as plt

print("=== BAGIAN 1: DATA TERSTRUKTUR (CSV) ===")

# Baca file CSV
df = pd.read_csv('mahasiswa.csv')

# Tampilkan info dasar
print("Info Dataset:")
print(f"Jumlah baris: {len(df)}")
print(f"Jumlah kolom: {len(df.columns)}")
print(f"Kolom: {list(df.columns)}")
print("\nData Mahasiswa:")
print(df)

# Statistik sederhana
print(f"\nRata-rata IPK: {df['ipk'].mean():.2f}")
print(f"IPK tertinggi: {df['ipk'].max()}")
print(f"IPK terendah: {df['ipk'].min()}")

# Mahasiswa dengan IPK tertinggi
mahasiswa_terbaik = df[df['ipk'] == df['ipk'].max()]
print(f"\nMahasiswa dengan IPK tertinggi:")
print(f"Nama: {mahasiswa_terbaik['nama'].values[0]}")
print(f"IPK: {mahasiswa_terbaik['ipk'].values[0]}")

# Visualisasi sederhana
plt.figure(figsize=(8, 5))
plt.bar(df['nama'], df['ipk'])
plt.title('IPK Mahasiswa')
plt.xlabel('Nama Mahasiswa')
plt.ylabel('IPK')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("=== SELESAI ===")