# ============================================
# stats_step8_viz_corr.py
# Visualisasi Correlation dengan Heatmap
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
np.random.seed(42)
n = 100

data = {
    'Study_Hours': np.random.uniform(1, 10, n),
    'Attendance': np.random.uniform(50, 100, n),
    'Sleep_Hours': np.random.uniform(4, 9, n),
    'Stress_Level': np.random.uniform(1, 10, n)
}

# Create correlations
data['Exam_Score'] = (
    5 * data['Study_Hours'] + 
    0.3 * data['Attendance'] + 
    2 * data['Sleep_Hours'] - 
    3 * data['Stress_Level'] + 
    np.random.normal(0, 5, n)
)

df = pd.DataFrame(data)

print("=" * 60)
print("CORRELATION MATRIX VISUALIZATION")
print("=" * 60)

# Calculate correlation matrix
corr_matrix = df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=axes[0], cbar_kws={"shrink": 0.8})
axes[0].set_title('Correlation Heatmap', fontsize=12, fontweight='bold')

# Pairplot style scatter
axes[1].scatter(df['Study_Hours'], df['Exam_Score'], alpha=0.5)
axes[1].set_xlabel('Study Hours')
axes[1].set_ylabel('Exam Score')
r_study = df['Study_Hours'].corr(df['Exam_Score'])
axes[1].set_title(f'Study Hours vs Exam Score (r={r_study:.3f})', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Top correlations dengan Exam_Score
print("\n" + "=" * 60)
print("CORRELATIONS WITH EXAM SCORE:")
print("=" * 60)
exam_corr = corr_matrix['Exam_Score'].sort_values(ascending=False)
for var, r in exam_corr.items():
    if var != 'Exam_Score':
        print(f"{var:20s}: r = {r:6.3f}")