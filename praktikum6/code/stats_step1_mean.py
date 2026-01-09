# ============================================
# stats_step1_mean.py
# Central Tendency: Mean
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data: Gaji karyawan (dalam juta rupiah/tahun)
salaries = [30, 35, 40, 45, 200]

print("=" * 60)
print("CENTRAL TENDENCY: MEAN")
print("=" * 60)
print(f"Salaries: {salaries}")

# Hitung mean
mean_salary = np.mean(salaries)
print(f"\nMean salary: Rp {mean_salary:.2f} juta/tahun")

# Apakah representatif?
print(f"\nApakah mean representatif?")
for sal in salaries:
    diff = abs(sal - mean_salary)
    print(f"  Rp {sal} juta → deviasi: Rp {diff:.2f} juta")

print(f"\nKesimpulan: Mean = Rp {mean_salary:.2f} juta")
print("Tidak representatif karena outlier (Rp 200 juta)")

# Visualisasi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(len(salaries)), salaries, color='skyblue', edgecolor='black')
plt.axhline(y=mean_salary, color='r', linestyle='--', label=f'Mean: {mean_salary:.1f}')
plt.ylabel('Salary (juta/tahun)')
plt.xlabel('Employee')
plt.title('Salaries with Mean')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(salaries, bins=10, color='lightcoral', edgecolor='black')
plt.axvline(x=mean_salary, color='r', linestyle='--', label=f'Mean: {mean_salary:.1f}')
plt.xlabel('Salary (juta/tahun)')
plt.ylabel('Frequency')
plt.title('Salary Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('figs/mean_example.png', dpi=300, bbox_inches='tight')
plt.show()