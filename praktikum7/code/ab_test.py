# ============================================
# ab_test.py
# A/B Testing: Uji Dua Proporsi
# ============================================

import numpy as np
from scipy import stats

print("=" * 60)
print("A/B TESTING: UJI WARNA BUTTON")
print("=" * 60)

# Data
n_A = 1000  # visitors varian A
x_A = 100   # konversi varian A
n_B = 1000  # visitors varian B
x_B = 130   # konversi varian B

p_A = x_A / n_A
p_B = x_B / n_B

alpha = 0.05

print(f"\nVarian A (Kontrol - Button Biru):")
print(f"  Visitors: {n_A}")
print(f"  Konversi: {x_A}")
print(f"  Conversion rate: {p_A:.1%}")

print(f"\nVarian B (Treatment - Button Merah):")
print(f"  Visitors: {n_B}")
print(f"  Konversi: {x_B}")
print(f"  Conversion rate: {p_B:.1%}")

print(f"\nPerbedaan: {(p_B - p_A):.1%}")
print(f"Alpha: {alpha}")

# Hipotesis
print("\n" + "=" * 60)
print("HIPOTESIS")
print("=" * 60)
print("H0: p_A = p_B (tidak ada perbedaan)")
print("H1: p_B > p_A (button merah lebih baik)")

# Two-proportion z-test
print("\n" + "=" * 60)
print("TWO-PROPORTION Z-TEST")
print("=" * 60)

# Pooled proportion
p_pool = (x_A + x_B) / (n_A + n_B)
print(f"Pooled proportion: {p_pool:.4f}")

# Standard error
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
print(f"Standard error: {se:.4f}")

# Z-statistic
z_stat = (p_B - p_A) / se
print(f"Z-statistic: {z_stat:.2f}")

# P-value (one-tailed, karena H1: p_B > p_A)
p_value = 1 - stats.norm.cdf(z_stat)
print(f"P-value (one-tailed): {p_value:.4f}")

# Keputusan
print("\n" + "=" * 60)
print("KEPUTUSAN")
print("=" * 60)

if p_value < alpha:
    print(f"P-value ({p_value:.4f}) < alpha ({alpha})")
    print("→ TOLAK H0")
    print("\nKesimpulan: Button merah SIGNIFIKAN meningkatkan konversi!")
    print(f"Peningkatan: {(p_B - p_A):.1%} (dari {p_A:.1%} ke {p_B:.1%})")
else:
    print(f"P-value ({p_value:.4f}) >= alpha ({alpha})")
    print("→ GAGAL TOLAK H0")
    print("\nKesimpulan: Tidak cukup bukti button merah lebih baik")

# Confidence interval untuk perbedaan proporsi
print("\n" + "=" * 60)
print("95% CI UNTUK PERBEDAAN PROPORSI")
print("=" * 60)

diff = p_B - p_A
se_diff = np.sqrt(p_A * (1 - p_A) / n_A + p_B * (1 - p_B) / n_B)
z_critical = 1.96
ci_lower = diff - z_critical * se_diff
ci_upper = diff + z_critical * se_diff

print(f"Perbedaan: {diff:.3f} atau {diff:.1%}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"        atau [{ci_lower:.1%}, {ci_upper:.1%}]")

if ci_lower > 0:
    print("\nCI tidak mengandung 0 → Button merah signifikan lebih baik")
else:
    print("\nCI mengandung 0 → Tidak signifikan")

# Effect size (relative lift)
print("\n" + "=" * 60)
print("EFFECT SIZE")
print("=" * 60)

relative_lift = (p_B - p_A) / p_A
print(f"Relative lift: {relative_lift:.1%}")
print(f"Artinya: Button merah meningkatkan konversi {relative_lift:.1%}")
print(f"         dibanding button biru")

# Praktis significance?
print("\n" + "=" * 60)
print("SIGNIFIKANSI PRAKTIS")
print("=" * 60)
print("Pertanyaan untuk bisnis:")
print(f"  - Apakah peningkatan {diff:.1%} worth it?")
print(f"  - Berapa revenue impact dari {diff:.1%} peningkatan?")
print(f"  - Ada biaya implementasi perubahan warna?")
print("\nStatistical significance ≠ Practical significance!")