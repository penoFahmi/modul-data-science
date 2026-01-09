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


# ============================================
# stats_step9_simpsons.py
# Simpson's Paradox
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("SIMPSON'S PARADOX")
print("=" * 60)

# Contoh: UC Berkeley Admission (simplified)
data = {
    'Department': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Applicants': [825, 108, 560, 25, 325, 593, 417, 375],
    'Admitted': [512, 89, 353, 17, 120, 202, 138, 131]
}

df = pd.DataFrame(data)
df['Admission_Rate'] = (df['Admitted'] / df['Applicants'] * 100).round(1)

print("\nData:")
print(df)

# Overall admission rate
overall = df.groupby('Gender').agg({
    'Applicants': 'sum',
    'Admitted': 'sum'
})
overall['Admission_Rate'] = (overall['Admitted'] / overall['Applicants'] * 100).round(1)

print("\n" + "=" * 60)
print("OVERALL ADMISSION RATE:")
print("=" * 60)
print(overall)

print(f"\nMale: {overall.loc['Male', 'Admission_Rate']:.1f}%")
print(f"Female: {overall.loc['Female', 'Admission_Rate']:.1f}%")
print(f"→ Terlihat bias terhadap Male!")

# Per department
print("\n" + "=" * 60)
print("ADMISSION RATE PER DEPARTMENT:")
print("=" * 60)
for dept in df['Department'].unique():
    dept_data = df[df['Department'] == dept]
    print(f"\nDepartment {dept}:")
    for _, row in dept_data.iterrows():
        print(f"  {row['Gender']:6s}: {row['Admission_Rate']:5.1f}%")

print("\n" + "=" * 60)
print("PARADOX:")
print("=" * 60)
print("Overall: Male admission > Female")
print("Per Dept: Female admission ≥ Male di hampir semua dept!")
print("\nPenyebab: Female apply ke competitive dept (low rate)")

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall
overall_plot = overall.reset_index()
axes[0].bar(overall_plot['Gender'], overall_plot['Admission_Rate'], 
            color=['lightblue', 'lightpink'], edgecolor='black')
axes[0].set_ylabel('Admission Rate (%)')
axes[0].set_title('Overall Admission Rate\n(Simpson\'s Paradox)', fontweight='bold')
axes[0].set_ylim(0, 50)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(overall_plot['Admission_Rate']):
    axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Per department
dept_pivot = df.pivot(index='Department', columns='Gender', values='Admission_Rate')
dept_pivot.plot(kind='bar', ax=axes[1], color=['lightblue', 'lightpink'], edgecolor='black')
axes[1].set_ylabel('Admission Rate (%)')
axes[1].set_xlabel('Department')
axes[1].set_title('Admission Rate per Department', fontweight='bold')
axes[1].legend(title='Gender')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figs/simpsons_paradox.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# stats_case_study.py
# Case Study: Student Performance Analysis
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate synthetic student data
np.random.seed(42)
n_students = 100

study_hours = np.random.uniform(5, 30, n_students)
attendance = np.random.uniform(60, 100, n_students)
exam_scores = (
    1.5 * study_hours + 
    0.3 * attendance + 
    np.random.normal(0, 8, n_students)
)
exam_scores = np.clip(exam_scores, 40, 100)  # Limit 40-100

df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Exam_Score': exam_scores
})

print("=" * 60)
print("CASE STUDY: STUDENT PERFORMANCE ANALYSIS")
print("=" * 60)
print(f"Number of students: {n_students}")
print(f"\nDataset preview:")
print(df.head(10))

# Descriptive statistics
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe().round(2))

# Central tendency
print("\n" + "=" * 60)
print("CENTRAL TENDENCY - EXAM SCORES")
print("=" * 60)
print(f"Mean: {df['Exam_Score'].mean():.2f}")
print(f"Median: {df['Exam_Score'].median():.2f}")
print(f"Mode: {stats.mode(df['Exam_Score'].round(), keepdims=True).mode[0]:.2f}")

# Dispersion
print("\n" + "=" * 60)
print("DISPERSION - EXAM SCORES")
print("=" * 60)
print(f"Range: {df['Exam_Score'].max() - df['Exam_Score'].min():.2f}")
print(f"Variance: {df['Exam_Score'].var():.2f}")
print(f"Std Dev: {df['Exam_Score'].std():.2f}")
print(f"IQR: {df['Exam_Score'].quantile(0.75) - df['Exam_Score'].quantile(0.25):.2f}")

# Correlation analysis
print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)
corr_matrix = df.corr()
print(corr_matrix.round(3))

r_study, p_study = stats.pearsonr(df['Study_Hours'], df['Exam_Score'])
r_attend, p_attend = stats.pearsonr(df['Attendance'], df['Exam_Score'])

print(f"\nStudy Hours vs Exam Score:")
print(f"  r = {r_study:.4f}, p-value = {p_study:.6f}")
print(f"  Interpretation: {'Strong' if abs(r_study) > 0.7 else 'Moderate' if abs(r_study) > 0.3 else 'Weak'} positive correlation")

print(f"\nAttendance vs Exam Score:")
print(f"  r = {r_attend:.4f}, p-value = {p_attend:.6f}")
print(f"  Interpretation: {'Strong' if abs(r_attend) > 0.7 else 'Moderate' if abs(r_attend) > 0.3 else 'Weak'} positive correlation")

# Insights
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"1. Average exam score: {df['Exam_Score'].mean():.1f} (median: {df['Exam_Score'].median():.1f})")
print(f"2. Scores vary by ±{df['Exam_Score'].std():.1f} points (moderate spread)")
print(f"3. Study hours is a stronger predictor (r={r_study:.3f}) than attendance (r={r_attend:.3f})")
print(f"4. Students study an average of {df['Study_Hours'].mean():.1f} hours/week")
print(f"5. Average attendance rate: {df['Attendance'].mean():.1f}%")

# Visualisasi
fig = plt.figure(figsize=(15, 10))

# 1. Score distribution
ax1 = plt.subplot(2, 3, 1)
plt.hist(df['Exam_Score'], bins=20, color='skyblue', edgecolor='black')
plt.axvline(df['Exam_Score'].mean(), color='r', linestyle='--', label=f'Mean: {df["Exam_Score"].mean():.1f}')
plt.axvline(df['Exam_Score'].median(), color='g', linestyle='--', label=f'Median: {df["Exam_Score"].median():.1f}')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.title('Exam Score Distribution')
plt.legend()
plt.grid(alpha=0.3)

# 2. Boxplot
ax2 = plt.subplot(2, 3, 2)
bp = plt.boxplot(df['Exam_Score'], vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
plt.ylabel('Exam Score')
plt.title('Exam Score Boxplot')
plt.grid(alpha=0.3)

# 3. Study hours vs Score
ax3 = plt.subplot(2, 3, 3)
plt.scatter(df['Study_Hours'], df['Exam_Score'], alpha=0.6, s=50)
z = np.polyfit(df['Study_Hours'], df['Exam_Score'], 1)
p = np.poly1d(z)
plt.plot(df['Study_Hours'], p(df['Study_Hours']), "r--", alpha=0.8)
plt.xlabel('Study Hours/Week')
plt.ylabel('Exam Score')
plt.title(f'Study Hours vs Score (r={r_study:.3f})')
plt.grid(alpha=0.3)

# 4. Attendance vs Score
ax4 = plt.subplot(2, 3, 4)
plt.scatter(df['Attendance'], df['Exam_Score'], alpha=0.6, s=50, color='green')
z = np.polyfit(df['Attendance'], df['Exam_Score'], 1)
p = np.poly1d(z)
plt.plot(df['Attendance'], p(df['Attendance']), "r--", alpha=0.8)
plt.xlabel('Attendance (%)')
plt.ylabel('Exam Score')
plt.title(f'Attendance vs Score (r={r_attend:.3f})')
plt.grid(alpha=0.3)

# 5. Correlation heatmap
ax5 = plt.subplot(2, 3, 5)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix')

# 6. Summary stats table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
SUMMARY STATISTICS

Exam Scores:
  Mean: {df['Exam_Score'].mean():.2f}
  Median: {df['Exam_Score'].median():.2f}
  Std Dev: {df['Exam_Score'].std():.2f}
  Range: {df['Exam_Score'].min():.1f} - {df['Exam_Score'].max():.1f}

Correlations with Score:
  Study Hours: r = {r_study:.3f}
  Attendance: r = {r_attend:.3f}

Insights:
  • Study hours adalah predictor terkuat
  • Moderate spread dalam scores
  • Distribusi relatif symmetric
"""
ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', 
         verticalalignment='center')

plt.tight_layout()
plt.savefig('figs/case_study_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Analysis complete! Check figs/case_study_analysis.png")
print("=" * 60)