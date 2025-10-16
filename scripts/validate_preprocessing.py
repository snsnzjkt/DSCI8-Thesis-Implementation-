"""
Validation script for preprocessing pipeline.
Loads data/processed/processed_data.pkl and validates:
 - Z-score normalization (mean≈0, std≈1)
 - Imputation completeness (no NaN/Inf)
 - SMOTE balance check
 - Outlier checks (remaining extremes)
 - Temporal split check (chronological preservation)
Generates visualizations in results/preprocessing/ and a report.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PKL = ROOT / 'data' / 'processed' / 'processed_data.pkl'
OUT_DIR = ROOT / 'results' / 'preprocessing'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tolerances
MEAN_TOL = 1e-2
STD_TOL = 1e-2

report_lines = []

if not PROCESSED_PKL.exists():
    print(f"Processed file not found: {PROCESSED_PKL}")
    raise SystemExit(1)

with open(PROCESSED_PKL, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data.get('feature_names', [f'f{i}' for i in range(X_train.shape[1])])
label_encoder = data.get('label_encoder', None)

report_lines.append('Preprocessing Validation Report')
report_lines.append('='*60)
report_lines.append(f'Processed file: {PROCESSED_PKL}')
report_lines.append('')

# 1) Imputation completeness
report_lines.append('1) Imputation completeness')
train_nan = np.isnan(X_train).sum()
test_nan = np.isnan(X_test).sum()
train_inf = np.isinf(X_train).sum()
test_inf = np.isinf(X_test).sum()
train_total = X_train.size
test_total = X_test.size

report_lines.append(f'   X_train NaN count: {train_nan} / {train_total} ({train_nan/train_total*100:.6f}%)')
report_lines.append(f'   X_train Inf count: {train_inf} / {train_total} ({train_inf/train_total*100:.6f}%)')
report_lines.append(f'   X_test  NaN count: {test_nan} / {test_total} ({test_nan/test_total*100:.6f}%)')
report_lines.append(f'   X_test  Inf count: {test_inf} / {test_total} ({test_inf/test_total*100:.6f}%)')

# Check completeness threshold (>=99.7% non-NaN)
train_non_nan_pct = 100.0 * (1 - train_nan / train_total)
imputation_ok = train_non_nan_pct >= 99.7
report_lines.append(f'   Imputation completeness (train non-NaN): {train_non_nan_pct:.4f}% -> {"OK" if imputation_ok else "FAIL"}')
report_lines.append('')

# 2) Normalization stats (means/std)
report_lines.append('2) Normalization (Z-score) checks')
if isinstance(X_train, np.ndarray):
    means = np.nanmean(X_train, axis=0)
    stds = np.nanstd(X_train, axis=0, ddof=0)
else:
    means = np.nanmean(X_train, axis=0)
    stds = np.nanstd(X_train, axis=0, ddof=0)

mean_violations = np.where(np.abs(means) > MEAN_TOL)[0]
std_violations = np.where(np.abs(stds - 1.0) > STD_TOL)[0]

report_lines.append(f'   Features checked: {len(means)}')
report_lines.append(f'   Features with |mean| > {MEAN_TOL}: {len(mean_violations)}')
report_lines.append(f'   Features with |std-1| > {STD_TOL}: {len(std_violations)}')
report_lines.append('')

# Save mean/std distributions plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(means, bins=50, color='C0', alpha=0.8)
plt.title('Feature means (train)')
plt.xlabel('mean')
plt.ylabel('count')
plt.axvline(0, color='k', linestyle='--')

plt.subplot(1,2,2)
plt.hist(stds, bins=50, color='C1', alpha=0.8)
plt.title('Feature stds (train)')
plt.xlabel('std')
plt.ylabel('count')
plt.axvline(1.0, color='k', linestyle='--')
plt.tight_layout()
plt.savefig(OUT_DIR / 'normalization_stats.png', dpi=150)
plt.close()
report_lines.append(f'   Saved normalization stats: {OUT_DIR / "normalization_stats.png"}')

# 3) SMOTE balance check
report_lines.append('3) SMOTE / Class balance')
try:
    counts = np.bincount(y_train)
except Exception:
    counts = None

if counts is not None:
    report_lines.append(f'   y_train distribution: {counts}')
    # balance ratio: max/min
    ratios = None
    if counts.min() > 0:
        ratios = counts.astype(float) / counts.min()
        report_lines.append(f'   Class ratios (relative to smallest): {np.round(ratios,3).tolist()}')
        smote_ok = np.allclose(ratios, np.ones_like(ratios), rtol=0.05)
        report_lines.append(f'   SMOTE 1:1 target achieved (≈equal counts): {smote_ok}')
    else:
        report_lines.append('   Warning: some classes have zero samples')
else:
    report_lines.append('   Could not read y_train distribution')
report_lines.append('')

# Save class distribution plot
plt.figure(figsize=(6,4))
if counts is not None:
    sns.barplot(x=list(range(len(counts))), y=counts, palette='viridis')
    plt.xlabel('Class (encoded)')
    plt.ylabel('Count')
    plt.title('y_train class distribution')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'class_distribution.png', dpi=150)
    plt.close()
    report_lines.append(f'   Saved class distribution: {OUT_DIR / "class_distribution.png"}')
else:
    report_lines.append('   Class distribution plot skipped')

# 4) Outlier check (remaining extremes)
report_lines.append('4) Outlier / extreme value checks')
# Compute z-scores using train means/stds (avoid div0)
stds_safe = np.where(stds == 0, 1.0, stds)
z_scores = (X_train - means) / stds_safe
abs_z = np.abs(z_scores)
extreme_count = np.sum(abs_z > 6)
extreme_total = X_train.size
report_lines.append(f'   Values with |z| > 6: {extreme_count} / {extreme_total} ({extreme_count/extreme_total*100:.6f}%)')

# Save top features by extreme counts
feature_extreme_counts = np.sum(abs_z > 6, axis=0)
top_extreme_idx = np.argsort(-feature_extreme_counts)[:10]
for idx in top_extreme_idx:
    report_lines.append(f'      Feature {idx} ({feature_names[idx]}): {feature_extreme_counts[idx]} extremes')

plt.figure(figsize=(8,4))
plt.bar(range(len(feature_extreme_counts)), feature_extreme_counts, color='C2')
plt.xlabel('Feature index')
plt.ylabel('Extremes count (|z|>6)')
plt.title('Per-feature extreme counts')
plt.tight_layout()
plt.savefig(OUT_DIR / 'feature_extreme_counts.png', dpi=150)
plt.close()
report_lines.append(f'   Saved feature extreme counts: {OUT_DIR / "feature_extreme_counts.png"}')
report_lines.append('')

# 5) Temporal split check
report_lines.append('5) Temporal split verification')
# We cannot strictly verify chronological split because timestamps were not preserved in processed_data.
# Heuristic: check if data was split with train_test_split (randomized) by inspecting contiguous label runs in train vs test
report_lines.append('   Note: processed_data.pkl does not include timestamps. The pipeline used sklearn.model_selection.train_test_split with stratify, which does NOT preserve chronological order.')
report_lines.append('   Conclusion: Temporal split is NOT chronological. If you require temporal validation, re-run preprocessing to keep timestamps and perform a time-based split (e.g., first 70% by time -> train, last 30% -> test).')
report_lines.append('')

# Write report text file
report_path = OUT_DIR / 'preprocessing_report.txt'
with open(report_path, 'w', encoding='utf-8') as rf:
    rf.write('\n'.join(report_lines))

print('Validation complete. Report and plots saved to:', OUT_DIR)
print('Summary:')
for line in report_lines[:12]:
    print(line)

print('\nFull report path:', report_path)

if __name__ == '__main__':
    pass
