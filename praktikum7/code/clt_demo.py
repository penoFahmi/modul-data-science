# ============================================
# clt_demo.py
# Demonstrasi Central Limit Theorem
# ============================================

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("DEMONSTRASI CENTRAL LIMIT THEOREM")
print("=" * 60)

# Set random seed
np.random.seed(42)

# Populasi dengan distribusi uniform (TIDAK normal)
populasi = np.random.uniform(0, 100, 10000)

print("\nPopulasi: Distribusi Uniform (0 - 100)")
print(f"Mean populasi: {np.mean(populasi):.2f}")
print(f"Std dev populasi: {np.std(populasi):.2f}")

# Ambil banyak sample dan hitung rata-ratanya
sample_sizes = [5, 10, 30, 100]
n_samples = 1000

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot populasi asli
axes[0, 0].hist(populasi, bins=25, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Populasi Asli\n(Uniform Distribution)', fontweight='bold')
axes[0, 0].set_xlabel('Nilai')
axes[0, 0].set_ylabel('Frekuensi')

# Untuk setiap ukuran sample
for idx, n in enumerate(sample_sizes):
    # Ambil banyak sample dan hitung rata-ratanya
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.choice(populasi, size=n, replace=True)
        sample_means.append(np.mean(sample))
    
    # Plot distribusi rata-rata sample
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    
    axes[row, col].hist(sample_means, bins=30, edgecolor='black', alpha=0.7)
    axes[row, col].axvline(np.mean(sample_means), color='r', linestyle='--', 
                           linewidth=2, label=f'Mean: {np.mean(sample_means):.1f}')
    axes[row, col].set_title(f'Sample Means (n={n})', fontweight='bold')
    axes[row, col].set_xlabel('Rata-rata Sample')
    axes[row, col].set_ylabel('Frekuensi')
    axes[row, col].legend()
    
    print(f"\nUkuran sample n = {n}:")
    print(f"  Mean of sample means: {np.mean(sample_means):.2f}")
    print(f"  Std dev of sample means: {np.std(sample_means):.2f}")
    print(f"  Teoritis std dev: {np.std(populasi) / np.sqrt(n):.2f}")

plt.tight_layout()
plt.savefig('figs/clt_demo.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)
print("Meskipun populasi uniform (TIDAK normal),")
print("distribusi rata-rata sample mendekati NORMAL")
print("terutama saat n >= 30!")
print("\nGambar disimpan: figs/clt_demo.png")