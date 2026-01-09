# ============================================
# stats_step6_covariance.py
# Correlation: Covariance
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# Data: Study hours vs Exam score
study_hours = [2, 3, 4, 5, 6, 7, 8]
exam_scores = [60, 65, 70, 75, 80, 85, 90]

print("=" * 60)
print("CORRELATION: COVARIANCE")
print("=" * 60)
print(f"Study Hours: {study_hours}")
print(f"Exam Scores: {exam_scores}")

# Hitung covariance
cov_matrix = np.cov(study_hours, exam_scores)
covariance = cov_matrix[0, 1]

print(f"\nCovariance: {covariance:.2f}")

# Manual calculation
mean_hours = np.mean(study_hours)
mean_scores = np.mean(exam_scores)
print(f"\nMean study hours: {mean_hours:.2f}")
print(f"Mean exam score: {mean_scores:.2f}")

deviations_product = []
print(f"\nCalculation:")
for h, s in zip(study_hours, exam_scores):
    dev_h = h - mean_hours
    dev_s = s - mean_scores
    product = dev_h * dev_s
    deviations_product.append(product)
    print(f"  ({h} - {mean_hours:.1f}) × ({s} - {mean_scores:.1f}) = {product:.2f}")

manual_cov = sum(deviations_product) / (len(study_hours) - 1)
print(f"\nCovariance (manual): {manual_cov:.2f}")

print(f"\n" + "=" * 60)
print(f"INTERPRETASI:")
print(f"  Covariance = {covariance:.2f} (positive)")
print(f"  → Study hours dan exam scores bergerak bersama")
print(f"  → Jika study hours naik, exam scores juga naik")
print(f"\nProblem: Unit tergantung scale, sulit compare")

# Visualisasi
plt.figure(figsize=(10, 5))
plt.scatter(study_hours, exam_scores, s=100, color='blue', alpha=0.6, edgecolor='black')
plt.plot(study_hours, exam_scores, 'r--', alpha=0.5)
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title(f'Study Hours vs Exam Score (Covariance: {covariance:.2f})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figs/covariance_example.png', dpi=300, bbox_inches='tight')
plt.show()