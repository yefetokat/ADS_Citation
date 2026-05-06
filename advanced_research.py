"""
ADVANCED RESEARCH STUDIES FOR CITATION PREDICTION
Complementary analyses to extend the main CitationGuess.ipynb notebook
Run this script after running CitationGuess.ipynb main analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Assume X_train, X_test, y_train, y_test are loaded from CitationGuess.ipynb
# If running standalone, load preprocessed data
print("="*70)
print("ADVANCED RESEARCH STUDIES FOR CITATION PREDICTION")
print("="*70)

# ==============================================================================
# RESEARCH 1: FEATURE ABLATION STUDY
# ==============================================================================
print("\n[1/8] FEATURE ABLATION STUDY")
print("-" * 70)

ablation_results = {}

# Baseline model
print("Training baseline model with all features...")
xgb_baseline = XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_baseline.fit(X_train, y_train)
y_pred_baseline = xgb_baseline.predict(X_test)
r2_baseline = r2_score(np.expm1(y_test), np.expm1(y_pred_baseline))
ablation_results['All Features (395)'] = r2_baseline
print(f"Baseline R²: {r2_baseline:.4f}")

# Without text embeddings (only 11 numeric features)
print("Training model WITHOUT text embeddings (11 numeric features)...")
X_train_no_text = X_train[:, 384:]
X_test_no_text = X_test[:, 384:]
xgb_no_text = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_no_text.fit(X_train_no_text, y_train)
y_pred_no_text = xgb_no_text.predict(X_test_no_text)
r2_no_text = r2_score(np.expm1(y_test), np.expm1(y_pred_no_text))
ablation_results['Without Text Embeddings (11)'] = r2_no_text
print(f"No Text Embeddings R²: {r2_no_text:.4f} (drop: {r2_baseline - r2_no_text:.4f})")

# Without PageRank (394 features)
print("Training model WITHOUT PageRank (394 features)...")
pagerank_idx = 9  # Index of pagerank in numeric features
X_train_no_pr = np.delete(X_train, 384+pagerank_idx, axis=1)
X_test_no_pr = np.delete(X_test, 384+pagerank_idx, axis=1)
xgb_no_pr = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_no_pr.fit(X_train_no_pr, y_train)
y_pred_no_pr = xgb_no_pr.predict(X_test_no_pr)
r2_no_pr = r2_score(np.expm1(y_test), np.expm1(y_pred_no_pr))
ablation_results['Without PageRank (394)'] = r2_no_pr
print(f"No PageRank R²: {r2_no_pr:.4f} (drop: {r2_baseline - r2_no_pr:.4f})")

# Without temporal features (392 features: -aging, -growth_rate, -early_signal)
print("Training model WITHOUT temporal features (392 features)...")
temporal_indices = [5, 7, 10]
X_train_no_temp = np.delete(X_train, [384+i for i in temporal_indices], axis=1)
X_test_no_temp = np.delete(X_test, [384+i for i in temporal_indices], axis=1)
xgb_no_temp = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_no_temp.fit(X_train_no_temp, y_train)
y_pred_no_temp = xgb_no_temp.predict(X_test_no_temp)
r2_no_temp = r2_score(np.expm1(y_test), np.expm1(y_pred_no_temp))
ablation_results['Without Temporal (392)'] = r2_no_temp
print(f"No Temporal Features R²: {r2_no_temp:.4f} (drop: {r2_baseline - r2_no_temp:.4f})")

# Only text embeddings
print("Training model with ONLY text embeddings (384 features)...")
X_train_text_only = X_train[:, :384]
X_test_text_only = X_test[:, :384]
xgb_text_only = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_text_only.fit(X_train_text_only, y_train)
y_pred_text_only = xgb_text_only.predict(X_test_text_only)
r2_text_only = r2_score(np.expm1(y_test), np.expm1(y_pred_text_only))
ablation_results['Only Text Embeddings (384)'] = r2_text_only
print(f"Text Only R²: {r2_text_only:.4f}")

# Print summary
print("\n" + "="*70)
print("ABLATION STUDY SUMMARY (R² Scores):")
print("="*70)
for feature_set, r2 in ablation_results.items():
    drop = r2_baseline - r2 if r2 != r2_baseline else 0
    drop_pct = (drop / r2_baseline * 100) if r2 != r2_baseline else 0
    status = "✓ BASELINE" if r2 == r2_baseline else f"↓ {drop:.4f} ({drop_pct:.1f}%)"
    print(f"  {feature_set:.<40} {r2:.4f}  {status}")

# Visualize
plt.figure(figsize=(12, 6))
features_short = ['All\nFeatures\n(395)', 'No Text\n(11)', 'No\nPageRank\n(394)',
                  'No\nTemporal\n(392)', 'Text\nOnly\n(384)']
r2_scores = list(ablation_results.values())
colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(r2_scores))]
bars = plt.bar(range(len(r2_scores)), r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('Feature Ablation Study: Impact on Model Performance', fontsize=14, fontweight='bold')
plt.xticks(range(len(r2_scores)), features_short, fontsize=10)
plt.ylim([min(r2_scores) * 0.95, max(r2_scores) * 1.01])
plt.grid(axis='y', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/ablation_study.png", dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/ablation_study.png")
plt.close()

# ==============================================================================
# RESEARCH 2: K-FOLD CROSS-VALIDATION
# ==============================================================================
print("\n[2/8] K-FOLD CROSS-VALIDATION (5-Fold)")
print("-" * 70)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=42)

print("Running 5-fold cross-validation on training data...")
cv_scores = cross_val_score(xgb_cv, X_train, y_train, cv=kfold,
                            scoring='r2', n_jobs=-1)

print(f"\nCross-Validation R² Scores by Fold:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"\nMean CV R²: {cv_scores.mean():.4f}")
print(f"Std Dev:    {cv_scores.std():.4f}")
print(f"95% CI:     [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

# Visualize CV scores
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cv_scores)+1), cv_scores, 'o-', linewidth=2, markersize=10, color='#3498db')
plt.axhline(cv_scores.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
plt.fill_between(range(1, len(cv_scores)+1),
                 cv_scores.mean() - cv_scores.std(),
                 cv_scores.mean() + cv_scores.std(),
                 alpha=0.2, color='#e74c3c', label=f'±1 Std: {cv_scores.std():.4f}')
plt.xlabel('Fold', fontsize=12, fontweight='bold')
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
plt.xticks(range(1, len(cv_scores)+1))
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/cross_validation.png", dpi=300, bbox_inches='tight')
print("✓ Saved: plots/cross_validation.png")
plt.close()

# ==============================================================================
# RESEARCH 3: ERROR ANALYSIS & RESIDUALS
# ==============================================================================
print("\n[3/8] ERROR ANALYSIS & RESIDUALS")
print("-" * 70)

y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_baseline)
residuals = y_test_real - y_pred_real
abs_errors = np.abs(residuals)

print(f"Error Statistics:")
print(f"  MAE:                {mean_absolute_error(y_test_real, y_pred_real):.2f} citations")
print(f"  RMSE:               {np.sqrt(mean_squared_error(y_test_real, y_pred_real)):.2f} citations")
print(f"  Median Abs Error:   {np.median(abs_errors):.2f} citations")
print(f"  95th percentile:    {np.percentile(abs_errors, 95):.2f} citations")

# Categorize errors
high_error_mask = abs_errors > np.percentile(abs_errors, 90)
low_citation_mask = y_test_real < np.percentile(y_test_real, 25)
high_citation_mask = y_test_real > np.percentile(y_test_real, 75)

print(f"\nError Distribution:")
print(f"  High errors (>90th percentile): {high_error_mask.sum()} papers")
print(f"  Low citation papers with high error: {(high_error_mask & low_citation_mask).sum()}")
print(f"  High citation papers with high error: {(high_error_mask & high_citation_mask).sum()}")

# Visualize residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Residuals vs Predicted
axes[0, 0].scatter(y_pred_real, residuals, alpha=0.3, s=20)
axes[0, 0].axhline(0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Citations', fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontweight='bold')
axes[0, 0].set_title('Residuals vs Predicted Values')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Residual distribution
axes[0, 1].hist(residuals, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Residual Value', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Absolute errors by citation range
citation_bins = [0, 10, 50, 100, 500, 1000, 10000, 100000]
citation_labels = ['0-10', '10-50', '50-100', '100-500', '500-1K', '1K-10K', '10K+']
binned_errors = pd.cut(y_test_real, bins=citation_bins, labels=citation_labels)
error_by_range = pd.DataFrame({'range': binned_errors, 'error': abs_errors}).groupby('range')['error'].agg(['mean', 'count'])

axes[1, 0].bar(range(len(error_by_range)), error_by_range['mean'].values, alpha=0.7, color='#e74c3c', edgecolor='black')
axes[1, 0].set_ylabel('Mean Absolute Error', fontweight='bold')
axes[1, 0].set_xlabel('Citation Count Range', fontweight='bold')
axes[1, 0].set_title('MAE by Citation Count Range')
axes[1, 0].set_xticks(range(len(error_by_range)))
axes[1, 0].set_xticklabels(error_by_range.index, rotation=45)
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Q-Q plot (residuals normality)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Residual Normality Check')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/error_analysis.png", dpi=300, bbox_inches='tight')
print("✓ Saved: plots/error_analysis.png")
plt.close()

# ==============================================================================
# RESEARCH 4: HYPERPARAMETER TUNING (Limited Grid Search)
# ==============================================================================
print("\n[4/8] HYPERPARAMETER TUNING (Limited Grid Search)")
print("-" * 70)

# Limited search to save time (sample from training data)
sample_size = min(3000, len(X_train))
sample_indices = np.random.choice(len(X_train), sample_size, replace=False)

param_grid = {
    'max_depth': [5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
}

xgb_grid = XGBRegressor(n_estimators=400, colsample_bytree=0.8, random_state=42)
grid_search = GridSearchCV(xgb_grid, param_grid, cv=3, scoring='r2', verbose=0, n_jobs=-1)

print("Running GridSearchCV on sample (3000 papers, 3-fold CV)...")
grid_search.fit(X_train[sample_indices], y_train[sample_indices])

print(f"\nBest Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV R² Score: {grid_search.best_score_:.4f}")

# Train with best params on full training data
print("\nTraining final model with best parameters on full dataset...")
xgb_tuned = XGBRegressor(**grid_search.best_params_, n_estimators=400,
                         colsample_bytree=0.8, random_state=42)
xgb_tuned.fit(X_train, y_train)
y_pred_tuned = xgb_tuned.predict(X_test)

r2_tuned = r2_score(np.expm1(y_test), np.expm1(y_pred_tuned))
spearman_tuned = spearmanr(np.expm1(y_test), np.expm1(y_pred_tuned)).correlation

print(f"\nTuned Model Test Performance:")
print(f"  R²:        {r2_tuned:.4f} (original: {r2_baseline:.4f}, Δ: {(r2_tuned-r2_baseline)*100:+.2f}%)")
print(f"  Spearman:  {spearman_tuned:.7f}")

# Top parameter combinations
top_n = 5
top_results = pd.DataFrame(grid_search.cv_results_).nlargest(top_n, 'mean_test_score')
print(f"\nTop {top_n} Parameter Combinations:")
print(top_results[['param_max_depth', 'param_learning_rate', 'param_subsample', 'mean_test_score']].to_string())

# ==============================================================================
# RESEARCH 5: PREDICTION CONFIDENCE INTERVALS (Bootstrap)
# ==============================================================================
print("\n[5/8] PREDICTION CONFIDENCE INTERVALS (Bootstrap)")
print("-" * 70)

n_bootstrap = 50
print(f"Running {n_bootstrap} bootstrap iterations...")

bootstrap_predictions = []
for i in range(n_bootstrap):
    if (i+1) % 10 == 0:
        print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")

    # Sample with replacement
    sample_idx = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train[sample_idx]
    y_boot = y_train[sample_idx]

    xgb_boot = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_boot.fit(X_boot, y_boot)

    y_pred_boot = xgb_boot.predict(X_test)
    bootstrap_predictions.append(np.expm1(y_pred_boot))

bootstrap_predictions = np.array(bootstrap_predictions)

# Calculate confidence intervals
pred_mean = bootstrap_predictions.mean(axis=0)
pred_std = bootstrap_predictions.std(axis=0)
pred_ci_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
pred_ci_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)

print(f"\nBootstrap Confidence Interval Statistics:")
print(f"  Mean CI width: {(pred_ci_upper - pred_ci_lower).mean():.2f} citations")
print(f"  Median CI width: {np.median(pred_ci_upper - pred_ci_lower):.2f} citations")
print(f"  Mean prediction std: {pred_std.mean():.2f}")

# Show examples
print(f"\nExample predictions with 95% CI (first 5 test samples):")
for i in range(5):
    actual = y_test_real[i]
    mean = pred_mean[i]
    lower = pred_ci_lower[i]
    upper = pred_ci_upper[i]
    contains = "✓" if lower <= actual <= upper else "✗"
    print(f"  Sample {i+1}: Actual={actual:.0f}, Pred={mean:.0f} [{lower:.0f}, {upper:.0f}] {contains}")

# Visualize uncertainty
fig, ax = plt.subplots(figsize=(14, 6))

# Sort by actual citations for better visualization
sort_idx = np.argsort(y_test_real)[:200]  # Show first 200 sorted samples

x = np.arange(len(sort_idx))
ax.scatter(x, y_test_real[sort_idx], label='Actual', s=30, alpha=0.6, color='black', zorder=3)
ax.plot(x, pred_mean[sort_idx], label='Predicted Mean', color='#3498db', linewidth=2)
ax.fill_between(x, pred_ci_lower[sort_idx], pred_ci_upper[sort_idx],
                alpha=0.2, color='#3498db', label='95% Confidence Interval')

ax.set_xlabel('Sample Index (sorted by actual citations)', fontweight='bold')
ax.set_ylabel('Citation Count', fontweight='bold')
ax.set_title('Predictions with Bootstrap Confidence Intervals (first 200 samples)', fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/confidence_intervals.png", dpi=300, bbox_inches='tight')
print("✓ Saved: plots/confidence_intervals.png")
plt.close()

# ==============================================================================
# RESEARCH 6: DOMAIN-SPECIFIC MODELS (By Research Field)
# ==============================================================================
print("\n[6/8] DOMAIN-SPECIFIC MODELS (By Research Field)")
print("-" * 70)

# Need to reload field information from original dataframe
# This assumes you have access to the original 'df' with field information
try:
    field_performance = {}

    # Get field assignments for test set
    test_df = pd.DataFrame({
        'field': df[test_mask]['field'].values,
        'actual': y_test_real,
        'predicted': y_pred_baseline,
    })

    print(f"Testing model performance by research field...")
    for field in test_df['field'].unique()[:5]:  # Show top 5 fields
        field_mask = test_df['field'] == field
        field_data = test_df[field_mask]

        if len(field_data) > 10:  # Only fields with sufficient samples
            field_r2 = r2_score(field_data['actual'], field_data['predicted'])
            field_n = len(field_data)
            field_performance[field] = {'R2': field_r2, 'N': field_n}

    # Train separate models per field (for largest fields)
    print(f"\nTraining domain-specific models for largest fields...")
    domain_model_performance = {}

    for field in test_df['field'].unique()[:3]:  # Top 3 fields by size
        train_field_mask = df[train_mask]['field'] == field
        test_field_mask = test_df['field'] == field

        if train_field_mask.sum() > 50 and test_field_mask.sum() > 10:
            X_train_field = X_train[train_field_mask]
            y_train_field = y_train[train_field_mask]
            X_test_field = X_test[test_field_mask]
            y_test_field = y_test[test_field_mask]

            xgb_domain = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42)
            xgb_domain.fit(X_train_field, y_train_field)
            y_pred_domain = xgb_domain.predict(X_test_field)

            r2_domain = r2_score(np.expm1(y_test_field), np.expm1(y_pred_domain))
            domain_model_performance[field] = r2_domain

            print(f"  {field[:30]:.<30} R²={r2_domain:.4f} (N={X_test_field.shape[0]})")

    print("✓ Domain-specific model analysis complete")
except Exception as e:
    print(f"Note: Domain-specific analysis requires original field data: {str(e)[:50]}")

# ==============================================================================
# RESEARCH 7: ENSEMBLE METHODS
# ==============================================================================
print("\n[7/8] ENSEMBLE METHODS")
print("-" * 70)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("Training ensemble models...")

models_ensemble = {
    'XGBoost': xgb_baseline,
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
}

ensemble_results = {}
ensemble_predictions = {}

for name, model in models_ensemble.items():
    if name != 'XGBoost':  # XGBoost already trained
        print(f"  Training {name}...")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))
    ensemble_results[name] = r2
    ensemble_predictions[name] = np.expm1(y_pred)
    print(f"    {name:.<20} R²={r2:.4f}")

# Simple average ensemble
print("\nCreating simple averaging ensemble...")
ensemble_avg = np.mean([ensemble_predictions[m] for m in ensemble_predictions.keys()], axis=0)
r2_ensemble = r2_score(y_test_real, ensemble_avg)
ensemble_results['Average Ensemble'] = r2_ensemble
print(f"  Average Ensemble:  R²={r2_ensemble:.4f}")

# Weighted ensemble (weights by individual R² scores)
print("Creating weighted ensemble (by R² score)...")
weights = np.array([ensemble_results[m] for m in models_ensemble.keys()])
weights = weights / weights.sum()
ensemble_weighted = np.average([ensemble_predictions[m] for m in ensemble_predictions.keys()],
                              axis=0, weights=weights)
r2_weighted = r2_score(y_test_real, ensemble_weighted)
ensemble_results['Weighted Ensemble'] = r2_weighted
print(f"  Weighted Ensemble: R²={r2_weighted:.4f}")

print(f"\nEnsemble Comparison:")
for name, r2 in sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:.<30} {r2:.4f}")

# Visualize ensemble
plt.figure(figsize=(10, 6))
names = list(ensemble_results.keys())
r2_vals = list(ensemble_results.values())
colors_ens = ['#3498db' if 'Ensemble' not in n else '#2ecc71' for n in names]
plt.bar(names, r2_vals, color=colors_ens, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.ylabel('R² Score', fontweight='bold')
plt.title('Model & Ensemble Comparison', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("plots/ensemble_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: plots/ensemble_comparison.png")
plt.close()

# ==============================================================================
# RESEARCH 8: CITATION TRAJECTORY ANALYSIS (By Year)
# ==============================================================================
print("\n[8/8] CITATION TRAJECTORY ANALYSIS (By Publication Year)")
print("-" * 70)

try:
    # Analyze model performance by publication year in test set
    test_df_years = pd.DataFrame({
        'year': df[test_mask]['year'].values,
        'actual': y_test_real,
        'predicted': y_pred_baseline,
        'error': np.abs(y_test_real - y_pred_baseline),
    })

    print("Model performance by publication year:")
    year_performance = test_df_years.groupby('year').agg({
        'actual': ['mean', 'count'],
        'error': 'mean',
        'predicted': lambda x: r2_score(test_df_years.loc[x.index, 'actual'], x)
    }).round(4)

    print(year_performance)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    years = test_df_years['year'].unique()

    # Plot 1: Actual vs Predicted by year
    for year in sorted(years):
        year_mask = test_df_years['year'] == year
        axes[0].scatter(test_df_years[year_mask]['actual'],
                       test_df_years[year_mask]['predicted'],
                       label=f'{year}', alpha=0.6, s=40)

    min_val = min(test_df_years['actual'].min(), test_df_years['predicted'].min())
    max_val = max(test_df_years['actual'].max(), test_df_years['predicted'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Citations', fontweight='bold')
    axes[0].set_ylabel('Predicted Citations', fontweight='bold')
    axes[0].set_title('Predictions by Publication Year')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Error by year
    year_errors = test_df_years.groupby('year')['error'].mean()
    axes[1].bar(year_errors.index, year_errors.values, alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[1].set_xlabel('Publication Year', fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Error', fontweight='bold')
    axes[1].set_title('MAE by Publication Year')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("plots/trajectory_analysis.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/trajectory_analysis.png")
    plt.close()

except Exception as e:
    print(f"Note: Trajectory analysis requires year information: {str(e)[:50]}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("RESEARCH SUMMARY")
print("="*70)

summary_results = {
    'Baseline XGBoost R²': r2_baseline,
    'Best Tuned Model R²': r2_tuned if 'r2_tuned' in locals() else 'N/A',
    'CV Mean R²': cv_scores.mean() if 'cv_scores' in locals() else 'N/A',
    'Ensemble R²': r2_ensemble if 'r2_ensemble' in locals() else 'N/A',
}

print("\nKey Findings:")
for finding, value in summary_results.items():
    if isinstance(value, float):
        print(f"  • {finding}: {value:.4f}")
    else:
        print(f"  • {finding}: {value}")

print("\nGenerated Plots:")
print("  ✓ plots/ablation_study.png")
print("  ✓ plots/cross_validation.png")
print("  ✓ plots/error_analysis.png")
print("  ✓ plots/confidence_intervals.png")
print("  ✓ plots/ensemble_comparison.png")
print("  ✓ plots/trajectory_analysis.png")

print("\n" + "="*70)
print("Advanced research complete! Check plots/ directory for visualizations.")
print("="*70)
