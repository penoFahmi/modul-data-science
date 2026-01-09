# ============================================
# stats_step2_median.py
# Central Tendency: Median
# ============================================

import numpy as np
import matplotlib.pyplot as plt

salaries = [30, 35, 40, 45, 200]

print("=" * 60)
print("CENTRAL TENDENCY: MEDIAN")
print("=" * 60)
print(f"Salaries (unsorted): {salaries}")

# Sort data
sorted_salaries = sorted(salaries)
print(f"Salaries (sorted): {sorted_salaries}")

# Hitung median
median_salary = np.median(salaries)
print(f"\nMedian salary: Rp {median_salary:.2f} juta/tahun")

# Compare dengan mean
mean_salary = np.mean(salaries)
print(f"Mean salary: Rp {mean_salary:.2f} juta/tahun")
print(f"\nMedian vs Mean:")
print(f"  Median lebih representatif: Rp {median_salary:.0f} juta")
print(f"  Mean terdistorsi outlier: Rp {mean_salary:.0f} juta")

# Contoh data genap
salaries_even = [30, 35, 40, 45, 50, 200]
median_even = np.median(salaries_even)
print(f"\n\nContoh dengan data genap (n=6):")
print(f"Data: {salaries_even}")
print(f"Median: ({salaries_even[2]} + {salaries_even[3]}) / 2 = {median_even:.1f}")

# Visualisasi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(len(salaries)), sorted_salaries, color='lightgreen', edgecolor='black')
plt.axhline(y=median_salary, color='g', linestyle='--', linewidth=2, label=f'Median: {median_salary:.0f}')
plt.axhline(y=mean_salary, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_salary:.0f}')
plt.ylabel('Salary (juta/tahun)')
plt.xlabel('Employee (sorted)')
plt.title('Median vs Mean')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
box_data = plt.boxplot(salaries, vert=True, patch_artist=True)
box_data['boxes'][0].set_facecolor('lightblue')
plt.ylabel('Salary (juta/tahun)')
plt.title('Boxplot - Median shown')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figs/median_example.png', dpi=300, bbox_inches='tight')
plt.show()
