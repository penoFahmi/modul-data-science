# ============================================
# stats_step4_variance.py
# Dispersion: Variance
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# Dua dataset dengan mean sama tapi spread berbeda
data_low_variance = [48, 49, 50, 51, 52]
data_high_variance = [30, 40, 50, 60, 70]

print("=" * 60)
print("DISPERSION: VARIANCE")
print("=" * 60)

# Dataset 1
mean1 = np.mean(data_low_variance)
var1 = np.var(data_low_variance, ddof=1)  # sample variance
print(f"Data 1: {data_low_variance}")
print(f"Mean: {mean1:.2f}")
print(f"Variance: {var1:.2f}")

# Hitung manual
deviations1 = [(x - mean1) for x in data_low_variance]
squared_dev1 = [(x - mean1)**2 for x in data_low_variance]
print(f"\nDeviations from mean: {[f'{d:.2f}' for d in deviations1]}")
print(f"Squared deviations: {[f'{d:.2f}' for d in squared_dev1]}")
print(f"Sum of squared dev: {sum(squared_dev1):.2f}")
print(f"Variance: {sum(squared_dev1)/(len(data_low_variance)-1):.2f}")

# Dataset 2
print(f"\n" + "-" * 60)
mean2 = np.mean(data_high_variance)
var2 = np.var(data_high_variance, ddof=1)
print(f"Data 2: {data_high_variance}")
print(f"Mean: {mean2:.2f}")
print(f"Variance: {var2:.2f}")

print(f"\n" + "=" * 60)
print(f"COMPARISON:")
print(f"  Both datasets have mean = 50")
print(f"  Data 1 variance: {var1:.2f} (low spread)")
print(f"  Data 2 variance: {var2:.2f} (high spread)")

# Visualisasi
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(data_low_variance, [1]*len(data_low_variance), s=100, alpha=0.6, color='blue')
plt.axvline(x=mean1, color='r', linestyle='--', label=f'Mean: {mean1:.0f}')
plt.xlim(20, 80)
plt.ylim(0.5, 1.5)
plt.xlabel('Value')
plt.title(f'Low Variance ({var1:.2f})')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(data_high_variance, [1]*len(data_high_variance), s=100, alpha=0.6, color='red')
plt.axvline(x=mean2, color='r', linestyle='--', label=f'Mean: {mean2:.0f}')
plt.xlim(20, 80)
plt.ylim(0.5, 1.5)
plt.xlabel('Value')
plt.title(f'High Variance ({var2:.2f})')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figs/variance_example.png', dpi=300, bbox_inches='tight')
plt.show()
