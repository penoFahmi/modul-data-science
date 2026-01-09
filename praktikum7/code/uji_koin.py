# ============================================
# uji_koin.py
# Uji Hipotesis: Apakah Koin Adil?
# ============================================

import numpy as np
from scipy import stats

print("=" * 60)
print("UJI HIPOTESIS: APAKAH KOIN ADIL?")
print("=" * 60)

# Setup
n_flips = 100
n_heads = 60
alpha = 0.05

print(f"\nData:")
print(f"  Jumlah lemparan: {n_flips}")
print(f"  Jumlah kepala: {n_heads}")
print(f"  Proporsi kepala: {n_heads/n_flips:.2f}")
print(f"  Alpha: {alpha}")

# Hipotesis
print("\n" + "=" * 60)
print("HIPOTESIS")
print("=" * 60)
print("H0: p = 0.5 (koin adil)")
print("H1: p ≠ 0.5 (koin tidak adil)")

# Uji binomial (exact test)
print("\n" + "=" * 60)
print("METODE 1: BINOMIAL TEST (EXACT)")
print("=" * 60)

# PERBAIKAN: gunakan binomtest
result = stats.binomtest(n_heads, n=n_flips, p=0.5, alternative='two-sided')
p_value_binomial = result.pvalue
print(f"P-value: {p_value_binomial:.4f}")

if p_value_binomial < alpha:
    print(f"Keputusan: Tolak H0 (p = {p_value_binomial:.4f} < {alpha})")
    print("Kesimpulan: Ada bukti signifikan koin tidak adil")
else:
    print(f"Keputusan: Gagal tolak H0 (p = {p_value_binomial:.4f} >= {alpha})")
    print("Kesimpulan: Tidak cukup bukti untuk bilang koin tidak adil")

# Uji z (approximation)
print("\n" + "=" * 60)
print("METODE 2: Z-TEST (APPROXIMATION)")
print("=" * 60)

# Hitung z-score
p0 = 0.5  # proporsi null hypothesis
expected = n_flips * p0
std_error = np.sqrt(n_flips * p0 * (1 - p0))
z_score = (n_heads - expected) / std_error

print(f"Expected heads (jika H0 benar): {expected:.0f}")
print(f"Standard error: {std_error:.2f}")
print(f"Z-score: {z_score:.2f}")

# P-value (two-tailed)
p_value_z = 2 * (1 - stats.norm.cdf(abs(z_score)))
print(f"P-value: {p_value_z:.4f}")

if p_value_z < alpha:
    print(f"Keputusan: Tolak H0 (p = {p_value_z:.4f} < {alpha})")
    print("Kesimpulan: Ada bukti signifikan koin tidak adil")
else:
    print(f"Keputusan: Gagal tolak H0 (p = {p_value_z:.4f} >= {alpha})")
    print("Kesimpulan: Tidak cukup bukti untuk bilang koin tidak adil")

# Confidence Interval untuk proporsi
print("\n" + "=" * 60)
print("95% CONFIDENCE INTERVAL UNTUK PROPORSI")
print("=" * 60)

p_hat = n_heads / n_flips
z_critical = 1.96
ci_margin = z_critical * np.sqrt(p_hat * (1 - p_hat) / n_flips)
ci_lower = p_hat - ci_margin
ci_upper = p_hat + ci_margin

print(f"Proporsi sample: {p_hat:.2f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"\nApakah 0.5 dalam CI? {ci_lower <= 0.5 <= ci_upper}")

if ci_lower <= 0.5 <= ci_upper:
    print("Ya → Konsisten dengan H0 (koin adil)")
else:
    print("Tidak → Tidak konsisten dengan H0 (koin tidak adil)")