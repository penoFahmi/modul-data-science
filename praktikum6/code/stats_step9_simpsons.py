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