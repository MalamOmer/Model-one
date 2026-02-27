"""
run_model.py
============
Local runner for the Hybrid RF → ANN maternal mortality model.

Since the original dataset is not committed to the repository,
this script generates a realistic synthetic dataset (500 rows)
with the same schema (ME, MA, AC, MM) and runs the full pipeline:

  - Class balance check
  - RF baseline
  - ANN baseline
  - Hybrid RF → ANN with 5-Fold Cross-Validation
  - Confusion matrices and ROC curves saved to ./results/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless rendering (no display needed)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score
)

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# ── 1. Synthetic data generation ──────────────────────────────────────────
print("=" * 60)
print("Generating synthetic maternal mortality dataset …")
print("=" * 60)

rng = np.random.default_rng(42)
N = 500

ME = rng.integers(1, 6, size=N)          # education: 1 (none) – 5 (university)
MA = rng.integers(15, 46, size=N)        # age: 15–45
AC = rng.integers(0, 11, size=N)         # antenatal visits: 0–10

# Risk score: low education + extreme age + few visits → higher mortality
risk = (
    (6 - ME) * 0.4          # lower education = higher risk
    + (np.abs(MA - 27)) * 0.08   # ages far from 27 = higher risk
    + (10 - AC) * 0.25       # fewer visits = higher risk
    + rng.normal(0, 0.3, N)  # noise
)
risk_norm = (risk - risk.min()) / (risk.max() - risk.min())

# Map to 3 classes (0, 1, 2)
MM = np.where(risk_norm < 0.55, 0, np.where(risk_norm < 0.80, 1, 2))

df = pd.DataFrame({'ME': ME, 'MA': MA, 'AC': AC, 'MM': MM})
df.to_excel('maternal_moterlity_synthetic.xlsx', index=False)

print(f"\nDataset shape : {df.shape}")
print("\nClass distribution (MM):")
print(df['MM'].value_counts().sort_index())
print(f"\nClass proportions:\n{df['MM'].value_counts(normalize=True).round(3).sort_index()}")

# ── 2. Binarize target ────────────────────────────────────────────────────
df['MM_binary'] = (df['MM'] >= 1).astype(int)
print(f"\nBinary target (0=no mortality, 1=mortality):\n"
      f"{df['MM_binary'].value_counts().sort_index()}")

# ── 3. Feature engineering ────────────────────────────────────────────────
def engineer_features(data):
    X = data[['ME', 'MA', 'AC']].copy()
    X['MA_squared']        = X['MA'] ** 2
    X['ME_AC_interaction'] = X['ME'] * X['AC']
    return X.values

X = engineer_features(df)
y = df['MM_binary'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples : {len(X_train)}  |  Test samples : {len(X_test)}")

# ── Helper: save confusion matrix ─────────────────────────────────────────
def save_cm(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ── Helper: save ROC curve ────────────────────────────────────────────────
def save_roc(fpr, tpr, roc_auc, label, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {label}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# A.  RANDOM FOREST BASELINE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("A.  Random Forest Classifier — Baseline")
print("=" * 60)

rf_base = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight='balanced'
)
rf_base.fit(X_train, y_train)

y_rf_pred  = rf_base.predict(X_test)
y_rf_proba = rf_base.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf_proba)
auc_rf = auc(fpr_rf, tpr_rf)
cm_rf  = confusion_matrix(y_test, y_rf_pred)
tn, fp, fn, tp = cm_rf.ravel()

print(classification_report(y_test, y_rf_pred,
                             target_names=['No Mortality', 'Mortality']))
print(f"AUC-ROC          : {auc_rf:.4f}")
print(f"False Negatives  : {fn}  (missed mortality cases — critical)")

print("\nFeature Importances:")
feat_names = ['ME', 'MA', 'AC', 'MA_squared', 'ME_AC_interaction']
for feat, imp in zip(feat_names, rf_base.feature_importances_):
    print(f"  {feat:20s}: {imp:.4f}")

save_cm(cm_rf, ['No Mortality', 'Mortality'],
        'Confusion Matrix — Random Forest',
        'results/cm_random_forest.png')
save_roc(fpr_rf, tpr_rf, auc_rf, 'Random Forest',
         'results/roc_random_forest.png')

# ══════════════════════════════════════════════════════════════════════════
# B.  ANN BASELINE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("B.  ANN (MLP) Classifier — Baseline")
print("=" * 60)

scaler_ann = StandardScaler().fit(X_train)
X_tr_s = scaler_ann.transform(X_train)
X_te_s = scaler_ann.transform(X_test)

ann_base = MLPClassifier(
    hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
)
ann_base.fit(X_tr_s, y_train)

y_ann_pred  = ann_base.predict(X_te_s)
y_ann_proba = ann_base.predict_proba(X_te_s)[:, 1]

fpr_ann, tpr_ann, _ = roc_curve(y_test, y_ann_proba)
auc_ann = auc(fpr_ann, tpr_ann)
cm_ann  = confusion_matrix(y_test, y_ann_pred)
tn2, fp2, fn2, tp2 = cm_ann.ravel()

print(classification_report(y_test, y_ann_pred,
                             target_names=['No Mortality', 'Mortality']))
print(f"AUC-ROC          : {auc_ann:.4f}")
print(f"False Negatives  : {fn2}  (missed mortality cases — critical)")

save_cm(cm_ann, ['No Mortality', 'Mortality'],
        'Confusion Matrix — ANN',
        'results/cm_ann.png')
save_roc(fpr_ann, tpr_ann, auc_ann, 'ANN',
         'results/roc_ann.png')

# ══════════════════════════════════════════════════════════════════════════
# C.  HYBRID RF → ANN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("C.  Hybrid RF → ANN")
print("=" * 60)

# Stage 1 — RF
rf_clf = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight='balanced'
)
rf_clf.fit(X_train, y_train)

rf_tr_p = rf_clf.predict_proba(X_train)[:, 1].reshape(-1, 1)
rf_te_p = rf_clf.predict_proba(X_test)[:, 1].reshape(-1, 1)

X_train_h = np.hstack([X_train, rf_tr_p])
X_test_h  = np.hstack([X_test,  rf_te_p])

# Scale
scaler = StandardScaler().fit(X_train_h)
X_tr_scaled = scaler.transform(X_train_h)
X_te_scaled = scaler.transform(X_test_h)

# Stage 2 — MLP
ann_clf = MLPClassifier(
    hidden_layer_sizes=(128, 128, 64),
    max_iter=1000,
    learning_rate='adaptive',
    random_state=42
)
ann_clf.fit(X_tr_scaled, y_train)

y_hyb_pred  = ann_clf.predict(X_te_scaled)
y_hyb_proba = ann_clf.predict_proba(X_te_scaled)[:, 1]

fpr_h, tpr_h, _ = roc_curve(y_test, y_hyb_proba)
auc_h  = auc(fpr_h, tpr_h)
cm_hyb = confusion_matrix(y_test, y_hyb_pred)
tn3, fp3, fn3, tp3 = cm_hyb.ravel()

print(classification_report(y_test, y_hyb_pred,
                             target_names=['No Mortality', 'Mortality']))
print(f"AUC-ROC          : {auc_h:.4f}")
print(f"False Negatives  : {fn3}  (missed mortality cases — critical)")

save_cm(cm_hyb, ['No Mortality', 'Mortality'],
        'Confusion Matrix — Hybrid RF→ANN',
        'results/cm_hybrid.png')
save_roc(fpr_h, tpr_h, auc_h, 'Hybrid RF→ANN',
         'results/roc_hybrid.png')

# Actual vs Predicted probability
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_hyb_proba, alpha=0.5)
ax.plot([0, 1], [0, 1], 'r-', label='Ideal')
ax.set_xlabel('Actual (0=No Mortality, 1=Mortality)')
ax.set_ylabel('Predicted Mortality Probability')
ax.set_title('Actual vs Predicted — Hybrid RF→ANN')
ax.legend()
plt.tight_layout()
plt.savefig('results/actual_vs_predicted_hybrid.png', dpi=150)
plt.close()

# 3-D scatter
fig3d = plt.figure(figsize=(7, 6))
ax3d  = fig3d.add_subplot(111, projection='3d')
sc = ax3d.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                  c=y_hyb_proba, cmap='viridis', alpha=0.7)
ax3d.set_xlabel('ME')
ax3d.set_ylabel('MA')
ax3d.set_zlabel('AC')
ax3d.set_title('Predicted Mortality Probability')
fig3d.colorbar(sc, pad=0.1, label='Mortality Probability')
plt.tight_layout()
plt.savefig('results/3d_scatter_hybrid.png', dpi=150)
plt.close()

# ── 5-Fold Cross-Validation ───────────────────────────────────────────────
print("\n" + "-" * 50)
print("5-Fold Stratified Cross-Validation — Hybrid RF→ANN")
print("-" * 50)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1, cv_auc = [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    rf_f = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight='balanced'
    )
    rf_f.fit(X_tr, y_tr)
    rf_tp = rf_f.predict_proba(X_tr)[:, 1].reshape(-1, 1)
    rf_vp = rf_f.predict_proba(X_val)[:, 1].reshape(-1, 1)

    X_trh = np.hstack([X_tr,  rf_tp])
    X_vlh = np.hstack([X_val, rf_vp])

    sc_f  = StandardScaler().fit(X_trh)
    X_trs = sc_f.transform(X_trh)
    X_vls = sc_f.transform(X_vlh)

    ann_f = MLPClassifier(
        hidden_layer_sizes=(128, 128, 64), max_iter=1000,
        learning_rate='adaptive', random_state=42
    )
    ann_f.fit(X_trs, y_tr)

    yv_pred  = ann_f.predict(X_vls)
    yv_proba = ann_f.predict_proba(X_vls)[:, 1]
    fpr_v, tpr_v, _ = roc_curve(y_val, yv_proba)

    fold_f1  = f1_score(y_val, yv_pred)
    fold_auc = auc(fpr_v, tpr_v)
    cv_f1.append(fold_f1)
    cv_auc.append(fold_auc)
    print(f"  Fold {fold + 1}: F1 = {fold_f1:.4f}  |  AUC = {fold_auc:.4f}")

print(f"\n  Mean F1  : {np.mean(cv_f1):.4f}  ±  {np.std(cv_f1):.4f}")
print(f"  Mean AUC : {np.mean(cv_auc):.4f}  ±  {np.std(cv_auc):.4f}")

# ── Profile prediction ────────────────────────────────────────────────────
def predict_profile(ME_v, MA_v, AC_v, rf_model, ann_model, fitted_scaler):
    profile = np.array([[ME_v, MA_v, AC_v, MA_v ** 2, ME_v * AC_v]])
    rf_proba = rf_model.predict_proba(profile)[:, 1].reshape(-1, 1)
    ph = np.hstack([profile, rf_proba])
    ps = fitted_scaler.transform(ph)
    return ann_model.predict_proba(ps)[0, 1]

examples = [
    (4, 28, 2),   # moderate risk
    (1, 16, 0),   # high risk (no education, teen, no visits)
    (5, 27, 8),   # low risk  (high education, prime age, many visits)
]
print("\n" + "-" * 50)
print("Profile Predictions")
print("-" * 50)
for ME_v, MA_v, AC_v in examples:
    p = predict_profile(ME_v, MA_v, AC_v, rf_clf, ann_clf, scaler)
    risk_label = 'HIGH' if p >= 0.5 else 'LOW'
    print(f"  ME={ME_v}, MA={MA_v}, AC={AC_v}  →  {p:.2%}  [{risk_label}]")

# ── Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'FN':>6}")
print("-" * 60)

for name, y_p, y_proba_v, cm_v in [
    ('Random Forest',     y_rf_pred,  y_rf_proba,  cm_rf),
    ('ANN (MLP)',         y_ann_pred, y_ann_proba, cm_ann),
    ('Hybrid RF→ANN',     y_hyb_pred, y_hyb_proba, cm_hyb),
]:
    acc = accuracy_score(y_test, y_p)
    f1  = f1_score(y_test, y_p)
    fpr_t, tpr_t, _ = roc_curve(y_test, y_proba_v)
    auc_t = auc(fpr_t, tpr_t)
    fn_t  = cm_v.ravel()[2]
    print(f"  {name:<23} {acc:>10.4f} {f1:>8.4f} {auc_t:>8.4f} {fn_t:>6}")

print("\nAll plots saved to ./results/")
print("Done.")
