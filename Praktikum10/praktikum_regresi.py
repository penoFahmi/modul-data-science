# ==========================================
# PART 1: SETUP & DATA EXPLORATION
# ==========================================

# Langkah 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Langkah 2: Load Dataset
# Load California Housing
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

print("Dataset Shape:", X.shape)
print("Features:", feature_names)
print("Target stats:")
print(f"Mean: {y.mean():.3f}")
print(f"Std: {y.std():.3f}")
print(f"Min: {y.min():.3f}")
print(f"Max: {y.max():.3f}")

# Langkah 3: Split Data
# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Langkah 4: Standardisasi Features
# PENTING! Regularisasi membutuhkan features yang terstandardisasi.
# Standardisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature means (scaled): ", X_train_scaled.mean(axis=0).round(3))
print("Feature stds (scaled): ", X_train_scaled.std(axis=0).round(3))


# ==========================================
# PART 2: OLS BASELINE
# ==========================================

# Fit OLS
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# Prediksi
y_train_pred_ols = ols.predict(X_train_scaled)
y_test_pred_ols = ols.predict(X_test_scaled)

# Evaluasi
train_rmse_ols = np.sqrt(mean_squared_error(y_train, y_train_pred_ols))
test_rmse_ols = np.sqrt(mean_squared_error(y_test, y_test_pred_ols))
train_r2_ols = r2_score(y_train, y_train_pred_ols)
test_r2_ols = r2_score(y_test, y_test_pred_ols)

print("=" * 60)
print("OLS (Baseline)")
print("=" * 60)
print(f"Training RMSE: {train_rmse_ols:.6f}")
print(f"Test RMSE: {test_rmse_ols:.6f}")
print(f"Training R^2: {train_r2_ols:.6f}")
print(f"Test R^2: {test_r2_ols:.6f}")
print(f"\nCoefficients magnitude (L2norm): {np.linalg.norm(ols.coef_):.4f}")
print(f"Max abs coefficient: {np.max(np.abs(ols.coef_)):.4f}")


# ==========================================
# PART 3: RIDGE REGRESSION
# ==========================================

# Implementasi - Manual Tuning
# Ridge dengan berbagai lambda
alphas_ridge = np.array([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
results_ridge = []

for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results_ridge.append({
        'alpha': alpha,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'coef_norm': np.linalg.norm(ridge.coef_)
    })

# Tampilkan hasil
ridge_df = pd.DataFrame(results_ridge)
print("\n" + "=" * 80)
print("RIDGE REGRESSION - Manual Tuning")
print("=" * 80)
print(ridge_df.to_string(index=False))

# Find best alpha
best_idx = ridge_df['test_rmse'].idxmin()
best_alpha_ridge = ridge_df.loc[best_idx, 'alpha']
best_test_rmse_ridge = ridge_df.loc[best_idx, 'test_rmse']

print(f"\nBest alpha: {best_alpha_ridge} (Test RMSE: {best_test_rmse_ridge:.6f})")

# Implementasi - RidgeCV (Automatic Tuning)
# RidgeCV automatic tuning
ridge_cv = RidgeCV(alphas=np.logspace(-2, 3, 100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)

optimal_alpha_ridge = ridge_cv.alpha_
print(f"\nRidgeCV Optimal alpha: {optimal_alpha_ridge:.6f}")

# Final model dengan optimal alpha
ridge_final = Ridge(alpha=optimal_alpha_ridge)
ridge_final.fit(X_train_scaled, y_train)

y_train_pred_ridge = ridge_final.predict(X_train_scaled)
y_test_pred_ridge = ridge_final.predict(X_test_scaled)

train_rmse_ridge = np.sqrt(mean_squared_error(y_train, y_train_pred_ridge))
test_rmse_ridge = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)

print(f"\nTraining RMSE: {train_rmse_ridge:.6f}")
print(f"Test RMSE: {test_rmse_ridge:.6f}")
print(f"Training R^2: {train_r2_ridge:.6f}")
print(f"Test R^2: {test_r2_ridge:.6f}")
print(f"Coefficient norm (shrinkage): {np.linalg.norm(ridge_final.coef_):.4f}")


# ==========================================
# PART 4: LASSO REGRESSION
# ==========================================

# Implementasi - Manual Tuning
# Lasso dengan berbagai lambda
alphas_lasso = np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0])
results_lasso = []

for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    y_train_pred = lasso.predict(X_train_scaled)
    y_test_pred = lasso.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    nonzero = np.count_nonzero(lasso.coef_)
    
    results_lasso.append({
        'alpha': alpha,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'nonzero_coef': nonzero,
        'coef_norm': np.linalg.norm(lasso.coef_)
    })

# Tampilkan hasil
lasso_df = pd.DataFrame(results_lasso)
print("\n" + "=" * 100)
print("LASSO REGRESSION - Manual Tuning")
print("=" * 100)
print(lasso_df.to_string(index=False))

# Find best alpha
best_idx = lasso_df['test_rmse'].idxmin()
best_alpha_lasso = lasso_df.loc[best_idx, 'alpha']
best_test_rmse_lasso = lasso_df.loc[best_idx, 'test_rmse']
print(f"\nBest alpha: {best_alpha_lasso} (Test RMSE: {best_test_rmse_lasso:.6f})")

# Implementasi - LassoCV (Automatic Tuning)
# LassoCV automatic tuning
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

optimal_alpha_lasso = lasso_cv.alpha_
print(f"\nLassoCV Optimal alpha: {optimal_alpha_lasso:.6f}")

# Final model dengan optimal alpha
lasso_final = Lasso(alpha=optimal_alpha_lasso, max_iter=10000)
lasso_final.fit(X_train_scaled, y_train)

y_train_pred_lasso = lasso_final.predict(X_train_scaled)
y_test_pred_lasso = lasso_final.predict(X_test_scaled)

train_rmse_lasso = np.sqrt(mean_squared_error(y_train, y_train_pred_lasso))
test_rmse_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))
train_r2_lasso = r2_score(y_train, y_train_pred_lasso)
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)
nonzero_lasso = np.count_nonzero(lasso_final.coef_)

print(f"\nTraining RMSE: {train_rmse_lasso:.6f}")
print(f"Test RMSE: {test_rmse_lasso:.6f}")
print(f"Training R^2: {train_r2_lasso:.6f}")
print(f"Test R^2: {test_r2_lasso:.6f}")
print(f"Non-zero coefficients: {nonzero_lasso}/{len(lasso_final.coef_)}")
print(f"Coefficient norm: {np.linalg.norm(lasso_final.coef_):.4f}")

# Feature Selection Analysis
# Analisis feature selection
print("\nFeature Selection (Non-zero coefficients):")
print("-" * 60)
for i, (name, coef) in enumerate(zip(feature_names, lasso_final.coef_)):
    if abs(coef) > 1e-10:
        print(f"{name:15s}: {coef:10.6f}")
    else:
        print(f"{name:15s}: ELIMINATED")


# ==========================================
# PART 5: ELASTIC NET
# ==========================================

# Implementasi
# ElasticNetCV dengan berbagai l1_ratio
l1_ratios = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]) # Updated to standard ratios, 0 and 1 handled by Ridge/Lasso logic usually
results_elasticnet = []

for l1_ratio in l1_ratios:
    elasticnet_cv = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=np.logspace(-3, 2, 50),
        cv=5,
        max_iter=10000
    )
    elasticnet_cv.fit(X_train_scaled, y_train)
    
    y_train_pred = elasticnet_cv.predict(X_train_scaled)
    y_test_pred = elasticnet_cv.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    nonzero = np.count_nonzero(elasticnet_cv.coef_)
    
    model_type = f"Elastic Net({l1_ratio})"
    
    results_elasticnet.append({
        'l1_ratio': l1_ratio,
        'model_type': model_type,
        'optimal_alpha': elasticnet_cv.alpha_,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'nonzero_coef': nonzero
    })

# Tampilkan hasil
elasticnet_df = pd.DataFrame(results_elasticnet)
print("\n" + "=" * 100)
print("ELASTIC NET - Ratio Comparison")
print("=" * 100)
print(elasticnet_df.to_string(index=False))

# Find best combination
best_idx = elasticnet_df['test_rmse'].idxmin()
best_l1_ratio = elasticnet_df.loc[best_idx, 'l1_ratio']
best_test_rmse_en = elasticnet_df.loc[best_idx, 'test_rmse']

print(f"\nBest L1 ratio: {best_l1_ratio} (Test RMSE: {best_test_rmse_en:.6f})")


# ==========================================
# PART 6: MODEL COMPARISON
# ==========================================

# Comparison table
comparison = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso', 'Elastic Net'],
    'Optimal Param': [
        '-', 
        f"Lambda={optimal_alpha_ridge:.4f}",
        f"Lambda={optimal_alpha_lasso:.4f}",
        f"Lambda={elasticnet_df.loc[best_idx, 'optimal_alpha']:.4f} (L1={best_l1_ratio})"
    ],
    'Train RMSE': [
        train_rmse_ols, 
        train_rmse_ridge,
        train_rmse_lasso, 
        elasticnet_df.loc[best_idx, 'train_rmse']
    ],
    'Test RMSE': [
        test_rmse_ols, 
        test_rmse_ridge,
        test_rmse_lasso, 
        elasticnet_df.loc[best_idx, 'test_rmse']
    ],
    'Train R^2': [
        train_r2_ols, 
        train_r2_ridge,
        train_r2_lasso, 
        elasticnet_df.loc[best_idx, 'train_r2']
    ],
    'Test R^2': [
        test_r2_ols, 
        test_r2_ridge,
        test_r2_lasso, 
        elasticnet_df.loc[best_idx, 'test_r2']
    ]
})

print("\n" + "=" * 120)
print("MODEL COMPARISON")
print("=" * 120)
print(comparison.to_string(index=False))

# Ranking
print("\n" + "=" * 60)
print("RANKING by Test RMSE (Lower is Better):")
print("=" * 60)
ranking = comparison.sort_values('Test RMSE').reset_index(drop=True)
for idx, row in ranking.iterrows():
    print(f"{idx+1}. {row['Model']:15s} - Test RMSE: {row['Test RMSE']:.6f}")


# ==========================================
# PART 7: ANALISIS & INTERPRETASI
# ==========================================

# Koefisien Comparison
# Bandingkan koefisien antar model
coef_comparison = pd.DataFrame({
    'Feature': feature_names,
    'OLS': ols.coef_,
    'Ridge': ridge_final.coef_,
    'Lasso': lasso_final.coef_
})

print("\n" + "=" * 100)
print("COEFFICIENT COMPARISON")
print("=" * 100)
print(coef_comparison.to_string(index=False))

# Normalization
print("\n" + "=" * 60)
print("COEFFICIENT MAGNITUDE (L2 Norm):")
print("=" * 60)
print(f"OLS:   {np.linalg.norm(ols.coef_):.6f}")
print(f"Ridge: {np.linalg.norm(ridge_final.coef_):.6f} (shrinkage)")
print(f"Lasso: {np.linalg.norm(lasso_final.coef_):.6f} (sparsity)")


# ==========================================
# PART 8: VISUALISASI (Optional)
# ==========================================

# Plot 1: Lambda vs RMSE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Ridge
ax1.plot(ridge_df['alpha'], ridge_df['train_rmse'], 'o-', label='Train', linewidth=2)
ax1.plot(ridge_df['alpha'], ridge_df['test_rmse'], 's-', label='Test', linewidth=2)
ax1.axvline(best_alpha_ridge, color='red', linestyle='--', label=f'Optimal Lambda {best_alpha_ridge:.4f}')
ax1.set_xlabel('Alpha', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('Ridge: RMSE vs Lambda', fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Lasso
ax2.plot(lasso_df['alpha'], lasso_df['train_rmse'], 'o-', label='Train', linewidth=2)
ax2.plot(lasso_df['alpha'], lasso_df['test_rmse'], 's-', label='Test', linewidth=2)
ax2.axvline(best_alpha_lasso, color='red', linestyle='--', label=f'Optimal Lambda {best_alpha_lasso:.4f}')
ax2.set_xlabel('Alpha', fontsize=12)
ax2.set_ylabel('RMSE', fontsize=12)
ax2.set_title('Lasso: RMSE vs Lambda', fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_lambda_vs_rmse.png', dpi=300, bbox_inches='tight')
print("\nSaved: 01_lambda_vs_rmse.png")
# plt.show() # Uncomment jika menggunakan Jupyter

# Plot 2: Model Comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = comparison['Model'].values
test_rmse_values = comparison['Test RMSE'].values
colors = ['steelblue', 'orange', 'green', 'red']

bars = ax.bar(models, test_rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Test RMSE', fontsize=12)
ax.set_title('Model Comparison: Test RMSE', fontsize=13, fontweight='bold')
ax.set_ylim([0, max(test_rmse_values) * 1.1])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('02_model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: 02_model_comparison.png")
# plt.show() # Uncomment jika menggunakan Jupyter