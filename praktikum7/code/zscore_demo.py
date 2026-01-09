# ============================================
# zscore_demo.py
# Demonstrasi Z-score
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 60)
print("DEMONSTRASI Z-SCORE")
print("=" * 60)

# Data: Nilai ujian mahasiswa
nilai_ujian = [55, 60, 65, 70, 75, 80, 85, 90, 95]

mean = np.mean(nilai_ujian)
std = np.std(nilai_ujian, ddof=1)

print(f"\nNilai ujian: {nilai_ujian}")
print(f"Mean: {mean:.2f}")
print(f"Std Dev: {std:.2f}")

# Hitung z-score untuk setiap nilai
print("\n" + "=" * 60)
print("Z-SCORE UNTUK SETIAP NILAI")
print("=" * 60)

for nilai in nilai_ujian:
    z = (nilai - mean) / std
    print(f"Nilai {nilai:2d}: z-score = {z:+.2f}", end="")
    
    if abs(z) > 2:
        print(" (Outlier!)")
    elif abs(z) > 1:
        print(" (Lumayan jauh dari mean)")
    else:
        print(" (Dekat dengan mean)")

# Contoh spesifik
print("\n" + "=" * 60)
print("CONTOH KASUS")
print("=" * 60)

mahasiswa_nilai = 85
z_mahasiswa = (mahasiswa_nilai - mean) / std

print(f"\nMahasiswa A mendapat nilai: {mahasiswa_nilai}")
print(f"Z-score: {z_mahasiswa:.2f}")
print(f"\nInterpretasi: Mahasiswa ini {abs(z_mahasiswa):.2f} std dev", end=" ")
print("di atas rata-rata" if z_mahasiswa > 0 else "di bawah rata-rata")

# Visualisasi
plt.figure(figsize=(12, 5))

# Histogram dengan mean line
plt.subplot(1, 2, 1)
plt.hist(nilai_ujian, bins=9, edgecolor='black', alpha=0.7)
plt.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
plt.axvline(mahasiswa_nilai, color='g', linestyle='-', linewidth=2, 
            label=f'Mahasiswa A: {mahasiswa_nilai}')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.title('Distribusi Nilai Ujian')
plt.legend()
plt.grid(alpha=0.3)

# Z-scores
plt.subplot(1, 2, 2)
z_scores = [(n - mean) / std for n in nilai_ujian]
plt.scatter(nilai_ujian, z_scores, s=100, alpha=0.6)
plt.axhline(0, color='r', linestyle='--', alpha=0.5)
plt.axhline(1, color='orange', linestyle=':', alpha=0.5)
plt.axhline(-1, color='orange', linestyle=':', alpha=0.5)
plt.axhline(2, color='red', linestyle=':', alpha=0.5)
plt.axhline(-2, color='red', linestyle=':', alpha=0.5)
plt.xlabel('Nilai Asli')
plt.ylabel('Z-score')
plt.title('Z-scores')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../figs/zscore_demo.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGambar disimpan: figs/zscore_demo.png")