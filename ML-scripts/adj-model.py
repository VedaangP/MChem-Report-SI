from ase.db import connect
import numpy as np
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Helper functions ───────────────────────────────────────────────────────────

def get_iodine_index(row) -> int:
    for idx, atomic_number in enumerate(row.numbers):
        if atomic_number == 53:
            return idx
    name = row.key_value_pairs.get("Name", f"id={row.id}")
    raise ValueError(f"No iodine atom found in molecule '{name}'")

def parse_db_array(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    elif isinstance(raw, np.ndarray):
        raw = raw.tolist()
    return np.array(raw)

# ── Scaffold ring partial charges (units: e) ──────────────────────────────────

SCAFFOLD_FACTORS = {
    "Benzene"    : -0.666233,
    "Pyrimidine" : -0.326117,
    "BX-5"       : -0.648058,
    "BX-6"       : -0.648042,
    "BZ-5"       : -0.642792,
    "BZ-6"       : -0.642793,
}

# ── Read database ──────────────────────────────────────────────────────────────

feat_db   = connect('mol.db')
target_db = connect('Cl-molecules.db')
features, targets, labels, scaffolds = [], [], [], []

interaction_e_lookup = {
    row.key_value_pairs.get("Name"): row.Interaction_E
    for row in target_db.select()
    if "Interaction_E" in row.key_value_pairs
}

for row in feat_db.select():
    name = row.key_value_pairs.get("Name")

    if name not in interaction_e_lookup:
        print(f" SKIP: '{name}' — no Interaction_E, skipping")
        continue

    i_index = get_iodine_index(row)

    monopole = parse_db_array(row.MWFN_MBIS_Atomic_Charges)[i_index]
    c6       = parse_db_array(row.MWFN_MBIS_c6)[i_index]
    dipole_0 = parse_db_array(row.MWFN_MBIS_Atom_Dipole)[i_index, 2]
    polar    = parse_db_array(row.MWFN_MBIS_Atomic_Polarizability)[i_index, 0]
    Q_0      = parse_db_array(row.MWFN_MBIS_Atom_Quadrupole)[i_index, 2]
    Q_2      = parse_db_array(row.MWFN_MBIS_Atom_Quadrupole)[i_index, 4]
    R2       = parse_db_array(row.MWFN_MBIS_r2)
    r2       = R2[i_index, 0] + R2[i_index, 1] + R2[i_index, 2]

    scaffold    = row.key_value_pairs.get("Aromtic_Scaffold", "unknown")
    ring_charge = SCAFFOLD_FACTORS.get(scaffold, 0.0)

    features.append([monopole, c6, dipole_0, r2, ring_charge])
    targets.append(interaction_e_lookup[name] * 627.5095)
    labels.append(name)
    scaffolds.append(scaffold)

print(f"Database read: {len(features)} molecules")

feature_names_num = ['Monopole', 'C6', 'Dipole_1', 'r2', 'Ring_Charge']
y     = np.array(targets)
label = np.array(labels)

# ── Build DataFrame ───────────────────────────────────────────────────────────

df = pd.DataFrame(features, columns=feature_names_num)
df['Scaffold'] = scaffolds

# ── 80/20 held-out split ──────────────────────────────────────────────────────

(df_train, df_test,
 y_train,  y_test,
 labels_train, labels_test) = train_test_split(
    df, y, label,
    test_size=0.2, random_state=42
)

print(f"Train: {len(df_train)} | Test: {len(df_test)}")

X_train = df_train[feature_names_num].values
X_test  = df_test[feature_names_num].values

# ── Pipeline: scale numericals, linear model ──────────────────────────────────

def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ])

pipeline = build_pipeline()

# ── K-Fold CV on training data ────────────────────────────────────────────────

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = -cross_val_score(
    pipeline, X_train, y_train,
    cv=kf, scoring='neg_root_mean_squared_error'
)
r2_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=kf, scoring='r2'
)

print("\n── Cross-Validation Results (training folds) ─────────────────────")
print(f"RMSE per fold : {rmse_scores.round(4)}")
print(f"Mean RMSE     : {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
print(f"R² per fold   : {r2_scores.round(4)}")
print(f"Mean R²       : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

# ── Per-fold weights ──────────────────────────────────────────────────────────

fold_weights = []
for train_idx, _ in kf.split(X_train):
    fp = build_pipeline()
    fp.fit(X_train[train_idx], y_train[train_idx])
    fold_weights.append(fp.named_steps['model'].coef_)

fold_weights = np.array(fold_weights)

print("\n── Feature Weights Across Folds ──────────────────────────────────")
for i, fname in enumerate(feature_names_num):
    print(f"{fname:>24}: {fold_weights[:, i].mean():.5f} ± {fold_weights[:, i].std():.5f}")

# ── Final model on full training set ─────────────────────────────────────────

pipeline.fit(X_train, y_train)

coefs = pipeline.named_steps['model'].coef_
bias  = pipeline.named_steps['model'].intercept_

print("\n── Descriptor Coefficients ───────────────────────────────────────")
for fname, c in zip(feature_names_num, coefs):
    print(f"{fname:>24}: {c:.5f}")
print(f"{'Bias':>24}: {bias:.5f}")

print("\n── Scaffold Ring Charges Used ────────────────────────────────────")
for scaffold, charge in sorted(SCAFFOLD_FACTORS.items()):
    print(f"{scaffold:>24}: {charge:+.4f} e")

# ── Training set predictions and metrics ──────────────────────────────────────

y_pred_train = pipeline.predict(X_train)
r2_train     = r2_score(y_train, y_pred_train)
rmse_train   = np.sqrt(mean_squared_error(y_train, y_pred_train))

# ── Held-out test set performance ─────────────────────────────────────────────

y_pred_test = pipeline.predict(X_test)
r2_test     = r2_score(y_test, y_pred_test)
rmse_test   = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n── Held-Out Test Set Performance ─────────────────────────────────")
print(f"{'R²':>24}: {r2_test:.4f}")
print(f"{'RMSE':>24}: {rmse_test:.4f} kcal/mol")

print("\n── Training Set Performance ──────────────────────────────────────")
print(f"{'R²':>24}: {r2_train:.4f}")
print(f"{'RMSE':>24}: {rmse_train:.4f} kcal/mol")

# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_interaction_energy(monopole, c6, dipole_0, r2, scaffold_name):
    """
    Predict interaction energy (kcal/mol) for a single molecule.
    scaffold_name is looked up in SCAFFOLD_FACTORS to get the ring charge;
    unknown scaffolds default to 0.0 e.
    """
    ring_charge = SCAFFOLD_FACTORS.get(scaffold_name, 0.0)
    X = np.array([[monopole, c6, dipole_0, r2, ring_charge]])
    return pipeline.predict(X)[0]

# ── Parity plot: train + test, coloured by scaffold ──────────────────────────

scaffolds_train = df_train['Scaffold'].to_numpy()
scaffolds_test  = df_test['Scaffold'].to_numpy()

all_scaffolds = sorted(set(scaffolds_train) | set(scaffolds_test))
palette = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#17becf',
]
color_map = {s: palette[i % len(palette)] for i, s in enumerate(all_scaffolds)}

fig, ax = plt.subplots(figsize=(7, 7))

# Training points — hollow markers
for scaffold in all_scaffolds:
    mask = scaffolds_train == scaffold
    if not mask.any():
        continue
    ax.scatter(
        y_train[mask], y_pred_train[mask],
        facecolors='grey',
        linewidths=1.2,
        s=55, zorder=2,
        label=f'Training Set',
    )

# Test points — filled markers with black edge
for scaffold in all_scaffolds:
    mask = scaffolds_test == scaffold
    if not mask.any():
        continue
    ax.scatter(
        y_test[mask], y_pred_test[mask],
        facecolors=color_map[scaffold],
        edgecolors='black',
        linewidths=0.5,
        s=55, zorder=3,
        label=f'{scaffold} (test)',
    )

# Perfect fit line
all_dft  = np.concatenate([y_train, y_test])
all_pred = np.concatenate([y_pred_train, y_pred_test])
pad      = 0.05 * (all_dft.max() - all_dft.min())
lims     = [all_dft.min() - pad, all_dft.max() + pad]
ax.plot(lims, lims, 'k--', linewidth=1.0, zorder=1)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_xlabel('DFT E_int (kcal/mol)', fontsize=13)
ax.set_ylabel('Predicted E_int (kcal/mol)', fontsize=13)

# R² and RMSE in the legend
metric_handles = [
    Line2D([], [], color='none',
           label=f'Test   R² = {r2_test:.3f}  |  RMSE = {rmse_test:.3f} kcal/mol'),
    Line2D([], [], color='none',
           label=f'Train  R² = {r2_train:.3f}  |  RMSE = {rmse_train:.3f} kcal/mol'),
    Line2D([], [], color='black', linestyle='--', linewidth=1.0,
           label='Perfect fit'),
]

scaffold_handles, scaffold_labels = ax.get_legend_handles_labels()
ax.legend(
    handles=metric_handles + scaffold_handles,
    labels=[h.get_label() for h in metric_handles] + scaffold_labels,
    fontsize=11,
    framealpha=0.9,
    loc='upper left',
)

fig.tight_layout()

for ext in ('pdf', 'svg', 'png'):
    path = f'parity_plot.{ext}'
    fig.savefig(path, dpi=300 if ext == 'png' else None)
    print(f'Parity plot saved → {path}')

plt.close(fig)
