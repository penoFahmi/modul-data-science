# ============================================
# stats_step5_stddev.py
# Dispersion: Standard Deviation
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# Dataset: Test scores
scores = [65, 70, 75, 80, 85, 90, 95]

print("=" * 60)
print("DISPERSION: STANDARD DEVIATION")
print("=" * 60)
print(f"Test scores: {scores}")

# Hitung statistik
mean_score = np.mean(scores)
var_score = np.var(scores, ddof=1)
std_score = np.std(scores, ddof=1)

print(f"\nMean: {mean_score:.2f}")
print(f"Variance: {var_score:.2f} (squared units - sulit interpretasi)")
print(f"Std Dev: {std_score:.2f} (same units as data)")

# Interpretasi
print(f"\n" + "=" * 60)
print(f"INTERPRETASI:")
print(f"  Typical score deviates ±{std_score:.2f} from mean")
print(f"  Range ±1 std: [{mean_score-std_score:.1f}, {mean_score+std_score:.1f}]")
print(f"  Range ±2 std: [{mean_score-2*std_score:.1f}, {mean_score+2*std_score:.1f}]")

# Hitung berapa data dalam ±1 std
within_1std = [s for s in scores if abs(s - mean_score) <= std_score]
pct_1std = len(within_1std) / len(scores) * 100
print(f"\n  {len(within_1std)}/{len(scores)} scores ({pct_1std:.1f}%) within ±1 std dev")

# Visualisasi
plt.figure(figsize=(10, 5))
plt.hist(scores, bins=7, color='lightblue', edgecolor='black', alpha=0.7)
plt.axvline(x=mean_score, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
plt.axvline(x=mean_score - std_score, color='orange', linestyle='--', linewidth=2, label=f'-1 SD: {mean_score-std_score:.1f}')
plt.axvline(x=mean_score + std_score, color='orange', linestyle='--', linewidth=2, label=f'+1 SD: {mean_score+std_score:.1f}')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title(f'Test Scores Distribution (Mean={mean_score:.1f}, SD={std_score:.1f})')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figs/stddev_example.png', dpi=300, bbox_inches='tight')
plt.show()
