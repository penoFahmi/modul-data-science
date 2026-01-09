# ============================================
# stats_step7_correlation.py
# Correlation: Pearson Correlation Coefficient
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data: Study hours vs Exam score
study_hours = [2, 3, 4, 5, 6, 7, 8]
exam_scores = [60, 65, 70, 75, 80, 85, 90]

print("=" * 60)
print("CORRELATION: PEARSON CORRELATION COEFFICIENT")
print("=" * 60)

# Hitung correlation
r, p_value = stats.pearsonr(study_hours, exam_scores)
print(f"Pearson r: {r:.4f}")
print(f"P-value: {p_value:.6f}")

# Interpretasi
print(f"\n" + "=" * 60)
print(f"INTERPRETASI:")
if abs(r) > 0.7:
    strength = "STRONG"
elif abs(r) > 0.3:
    strength = "MODERATE"
else:
    strength = "WEAK"

direction = "POSITIVE" if r > 0 else "NEGATIVE" if r < 0 else "NO"
print(f"  Correlation: {strength} {direction}")
print(f"  r = {r:.4f}")

if r > 0:
    print(f"  → Study hours naik → Exam scores naik")
elif r < 0:
    print(f"  → Study hours naik → Exam scores turun")

# Contoh berbagai korelasi
np.random.seed(42)
x = np.linspace(0, 10, 50)

# Perfect positive
y_perfect = 2 * x + 1

# Strong positive
y_strong = 2 * x + np.random.normal(0, 2, 50)

# Weak positive
y_weak = 2 * x + np.random.normal(0, 10, 50)

# No correlation
y_none = np.random.normal(10, 5, 50)

# Negative
y_negative = -2 * x + 20 + np.random.normal(0, 2, 50)

# Calculate correlations
r_perfect = np.corrcoef(x, y_perfect)[0, 1]
r_strong = np.corrcoef(x, y_strong)[0, 1]
r_weak = np.corrcoef(x, y_weak)[0, 1]
r_none = np.corrcoef(x, y_none)[0, 1]
r_negative = np.corrcoef(x, y_negative)[0, 1]

# Visualisasi
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Original data
axes[0, 0].scatter(study_hours, exam_scores, s=80, alpha=0.6, edgecolor='black')
axes[0, 0].set_title(f'Original Data\nr = {r:.3f}')
axes[0, 0].set_xlabel('Study Hours')
axes[0, 0].set_ylabel('Exam Score')
axes[0, 0].grid(alpha=0.3)

# Perfect positive
axes[0, 1].scatter(x, y_perfect, s=30, alpha=0.6)
axes[0, 1].set_title(f'Perfect Positive\nr = {r_perfect:.3f}')
axes[0, 1].grid(alpha=0.3)

# Strong positive
axes[0, 2].scatter(x, y_strong, s=30, alpha=0.6)
axes[0, 2].set_title(f'Strong Positive\nr = {r_strong:.3f}')
axes[0, 2].grid(alpha=0.3)

# Weak positive
axes[1, 0].scatter(x, y_weak, s=30, alpha=0.6)
axes[1, 0].set_title(f'Weak Positive\nr = {r_weak:.3f}')
axes[1, 0].grid(alpha=0.3)

# No correlation
axes[1, 1].scatter(x, y_none, s=30, alpha=0.6)
axes[1, 1].set_title(f'No Correlation\nr = {r_none:.3f}')
axes[1, 1].grid(alpha=0.3)

# Negative
axes[1, 2].scatter(x, y_negative, s=30, alpha=0.6)
axes[1, 2].set_title(f'Negative\nr = {r_negative:.3f}')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figs/correlation_examples.png', dpi=300, bbox_inches='tight')
plt.show()
