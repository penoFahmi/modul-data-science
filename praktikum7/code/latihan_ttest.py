# ============================================
# latihan_ttest.py
# Latihan: T-test untuk Satu Sample
# ============================================

import numpy as np
from scipy import stats

print("=" * 60)
print("LATIHAN: UJI T-TEST")
print("=" * 60)
print("\nSoal:")
print("Pabrik klaim baterai tahan 500 jam.")
print("Sample 25 baterai ditest:")
print("  - Rata-rata sample: 485 jam")
print("  - Std dev sample: 40 jam")
print("\nDengan alpha = 0.05, apakah ada bukti rata-rata berbeda dari klaim?")

# Data
n = 25
sample_mean = 485
sample_std = 40
claimed_mean = 500
alpha = 0.05

print("\n" + "=" * 60)
print("SETUP")
print("=" * 60)
print(f"n = {n}")
print(f"sample mean = {sample_mean}")
print(f"sample std = {sample_std}")
print(f"claimed mean (mu0) = {claimed_mean}")
print(f"alpha = {alpha}")

# Hipotesis
print("\n" + "=" * 60)
print("HIPOTESIS")
print("=" * 60)
print(f"H0: mu = {claimed_mean} (klaim benar)")
print(f"H1: mu ≠ {claimed_mean} (klaim salah)")

# T-test
print("\n" + "=" * 60)
print("ONE-SAMPLE T-TEST")
print("=" * 60)

# Hitung t-statistic
t_stat = (sample_mean - claimed_mean) / (sample_std / np.sqrt(n))
print(f"T-statistic: {t_stat:.2f}")

# Degrees of freedom
df = n - 1
print(f"Degrees of freedom: {df}")

# P-value (two-tailed)
p_value = 2 * stats.t.cdf(t_stat, df)  # karena t_stat negatif
print(f"P-value (two-tailed): {p_value:.4f}")

# Critical value
t_critical = stats.t.ppf(1 - alpha/2, df)
print(f"T-critical (alpha={alpha}): ±{t_critical:.3f}")

# Keputusan
print("\n" + "=" * 60)
print("KEPUTUSAN")
print("=" * 60)

print(f"|T-statistic| = {abs(t_stat):.2f}")
print(f"T-critical = {t_critical:.3f}")

if abs(t_stat) > t_critical:
    print(f"\n|T-stat| > T-critical → TOLAK H0")
    print(f"P-value ({p_value:.4f}) < alpha ({alpha})")
    print("\nKesimpulan: Ada bukti signifikan rata-rata lifetime")
    print(f"            berbeda dari klaim {claimed_mean} jam")
else:
    print(f"\n|T-stat| <= T-critical → GAGAL TOLAK H0")
    print(f"P-value ({p_value:.4f}) >= alpha ({alpha})")
    print("\nKesimpulan: Tidak cukup bukti untuk bilang klaim salah")

# 95% Confidence Interval
print("\n" + "=" * 60)
print("95% CONFIDENCE INTERVAL")
print("=" * 60)

se = sample_std / np.sqrt(n)
margin = t_critical * se
ci_lower = sample_mean - margin
ci_upper = sample_mean + margin

print(f"Standard error: {se:.2f}")
print(f"Margin of error: {margin:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

if ci_lower <= claimed_mean <= ci_upper:
    print(f"\n{claimed_mean} dalam CI → Konsisten dengan H0")
else:
    print(f"\n{claimed_mean} TIDAK dalam CI → Tidak konsisten dengan H0")

print("\n" + "=" * 60)
print("JAWABAN LENGKAP")
print("=" * 60)
print(f"1. H0: mu = {claimed_mean}, H1: mu ≠ {claimed_mean}")
print(f"2. T-statistic = {t_stat:.2f}")
print(f"3. P-value = {p_value:.4f}")
print(f"4. Keputusan: {'Tolak H0' if p_value < alpha else 'Gagal tolak H0'}")
print(f"5. Kesimpulan: {'Rata-rata berbeda dari klaim' if p_value < alpha else 'Tidak cukup bukti rata-rata berbeda'}")