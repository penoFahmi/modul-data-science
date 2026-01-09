# ============================================
# confidence_interval.py
# Confidence Interval untuk Mean
# ============================================

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

print("=" * 60)
print("CONFIDENCE INTERVAL UNTUK MEAN")
print("=" * 60)

# Data: Tinggi mahasiswa (cm)
tinggi = [165, 170, 168, 175, 172, 169, 180, 167, 173, 171,
          174, 169, 176, 168, 170, 172, 177, 165, 171, 173,
          169, 174, 170, 168, 175, 171, 173, 169, 172, 170,
          166, 174, 171, 169, 173, 172]

n = len(tinggi)
mean = np.mean(tinggi)
std = np.std(tinggi, ddof=1)

print(f"\nData: Tinggi {n} mahasiswa")
print(f"Mean: {mean:.2f} cm")
print(f"Std dev: {std:.2f} cm")

# Confidence levels
confidence_levels = [0.90, 0.95, 0.99]

print("\n" + "=" * 60)
print("CONFIDENCE INTERVALS")
print("=" * 60)

for conf in confidence_levels:
    # Degrees of freedom
    df = n - 1
    
    # Critical value dari t-distribution
    t_critical = stats.t.ppf((1 + conf) / 2, df)
    
    # Standard error
    se = std / np.sqrt(n)
    
    # Margin of error
    margin = t_critical * se
    
    # Confidence interval
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    print(f"\n{int(conf*100)}% Confidence Interval:")
    print(f"  t-critical (df={df}): {t_critical:.3f}")
    print(f"  Standard error: {se:.2f}")
    print(f"  Margin of error: {margin:.2f}")
    print(f"  CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  Interpretasi: Kita {int(conf*100)}% yakin tinggi rata-rata")
    print(f"                mahasiswa antara {ci_lower:.1f} dan {ci_upper:.1f} cm")

# Visualisasi
plt.figure(figsize=(12, 5))

# Histogram data
plt.subplot(1, 2, 1)
plt.hist(tinggi, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f} cm')
plt.xlabel('Tinggi (cm)')
plt.ylabel('Frekuensi')
plt.title('Distribusi Tinggi Mahasiswa')
plt.legend()
plt.grid(alpha=0.3)

# Confidence intervals
plt.subplot(1, 2, 2)
y_pos = [1, 2, 3]
colors = ['lightblue', 'lightgreen', 'lightcoral']

for i, conf in enumerate(confidence_levels):
    df = n - 1
    t_critical = stats.t.ppf((1 + conf) / 2, df)
    se = std / np.sqrt(n)
    margin = t_critical * se
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    plt.barh(y_pos[i], ci_upper - ci_lower, left=ci_lower, height=0.3, 
             color=colors[i], alpha=0.7, edgecolor='black',
             label=f'{int(conf*100)}% CI: [{ci_lower:.1f}, {ci_upper:.1f}]')

plt.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
plt.yticks(y_pos, ['90%', '95%', '99%'])
plt.xlabel('Tinggi (cm)')
plt.title('Confidence Intervals untuk Mean')
plt.legend(loc='lower right', fontsize=8)
plt.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../figs/confidence_interval.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGambar disimpan: figs/confidence_interval.png")