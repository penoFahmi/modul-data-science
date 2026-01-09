# ============================================
# stats_step3_mode.py
# Central Tendency: Mode
# ============================================

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Data: Rating produk
ratings = [5, 4, 5, 3, 5, 4, 5, 2, 4, 5]

print("=" * 60)
print("CENTRAL TENDENCY: MODE")
print("=" * 60)
print(f"Ratings: {ratings}")

# Hitung mode
mode_result = stats.mode(ratings, keepdims=True)
mode_value = mode_result.mode[0]
mode_count = mode_result.count[0]

print(f"\nMode: {mode_value} (muncul {mode_count} kali)")
print(f"Mean: {np.mean(ratings):.2f}")
print(f"Median: {np.median(ratings):.1f}")

# Frekuensi setiap nilai
unique, counts = np.unique(ratings, return_counts=True)
print(f"\nFrekuensi:")
for val, count in zip(unique, counts):
    print(f"  Rating {val}: {count} kali")

# Contoh categorical
sizes = ['S', 'M', 'M', 'L', 'M', 'XL', 'M', 'S', 'M']
print(f"\n\nContoh Categorical Data (Ukuran Baju):")
print(f"Sizes: {sizes}")
size_mode = max(set(sizes), key=sizes.count)
print(f"Mode: {size_mode} (ukuran paling populer)")

# Visualisasi
plt.figure(figsize=(12, 4))

# Rating distribution
plt.subplot(1, 2, 1)
plt.bar(unique, counts, color='gold', edgecolor='black', width=0.6)
plt.axvline(x=mode_value, color='r', linestyle='--', linewidth=2, label=f'Mode: {mode_value}')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Rating Distribution')
plt.xticks(unique)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Size distribution
plt.subplot(1, 2, 2)
size_unique, size_counts = np.unique(sizes, return_counts=True)
plt.bar(size_unique, size_counts, color='lightcoral', edgecolor='black')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.title('Product Size Distribution')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figs/mode_example.png', dpi=300, bbox_inches='tight')
plt.show()
