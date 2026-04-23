"""
gb-adj-model.py

Gradient Boosting Regressor with leave-one-acceptor-out test split.

Data loading:
  - Acceptor polarizability  : anion-molecules.db  (acc_polar per halogen)
  - Interaction energies     : I/Cl/Br/F-molecules.db
  - Iodine donor descriptors : mol.db  (monopole, c6, dipole_0, r2, ring_charge)

Split:
  - TEST_ACCEPTOR held out as blind test set (default: 'nan-Br')
  - All other acceptors used for training

Training:
  - Bayesian Optimisation (bayes_opt, Expected Improvement)
    with Latin Hypercube Sampling for initialisation
  - Repeated 5-Fold × 10 repeats, scored by RMSE
  - StandardScaler fit on train only
  - Seed: 0 (ROBERT default)
"""

from ase.db import connect
import numpy as np
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score, learning_curve, KFold
from sklearn.metrics import mean_squared_error, r2_score

from bayes_opt import BayesianOptimization
from scipy.stats import qmc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# =============================================================================
# CONFIGURATION
# Change TEST_ACCEPTOR to hold out a different acceptor for blind prediction.
# Valid options: 'nan-I', 'nan-F', 'nan-Cl', 'nan-Br'
# =============================================================================

TEST_ACCEPTOR = 'Br'

HALOGEN_DBS = {
    'I':  ('ASE-databases/I-molecules.db',  53),
    'F':  ('ASE-databases/F-molecules.db',   9),
    'Cl': ('ASE-databases/Cl-molecules.db', 17),
    'Br': ('ASE-databases/Br-molecules.db', 35),
}

FEAT_DB_PATH = 'ASE-databases/mol.db'
ACC_DB_PATH  = 'ASE-databases/anion-molecules.db'

SCAFFOLD_FACTORS = {
    "Benzene"   : -0.666233,
    "Pyrimidine": -0.326117,
    "BX-5"      : -0.648058,
    "BX-6"      : -0.648042,
    "BZ-5"      : -0.642792,
    "BZ-6"      : -0.642793,
}

SEED = 0

# -- Helper functions ---------------------------------------------------------

def get_iodine_index(row) -> int:
    for idx, z in enumerate(row.numbers):
        if z == 53:
            return idx
    raise ValueError(
        f"No iodine in '{row.key_value_pairs.get('Name', f'id={row.id}')}'")

def get_atom_index(row, atomic_number: int) -> int:
    for idx, z in enumerate(row.numbers):
        if z == atomic_number:
            return idx
    raise ValueError(f"Atom Z={atomic_number} not found in row id={row.id}")

def parse_db_array(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    elif isinstance(raw, np.ndarray):
        raw = raw.tolist()
    return np.array(raw)

def ensure_2d(arr, ncols):
    return arr.reshape(-1, ncols) if arr.ndim == 1 else arr

def save_fig(fig, stem):
    for ext in ("pdf", "svg", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"  Saved -> {path}")
    plt.close(fig)

# =============================================================================
# Step 1: Load acceptor polarizabilities from anion-molecules.db
# =============================================================================

print("=" * 60)
print("Step 1 - Acceptor polarizabilities")
print("=" * 60)

acc_db = connect(ACC_DB_PATH)
acc_polar_lookup = {}  # {acceptor_label: float}
acc_c6_lookup = {}

for row in acc_db.select():
    name = row.key_value_pairs.get("Name", "").strip()
    if name not in HALOGEN_DBS:
        continue
    _, acc_z = HALOGEN_DBS[name]
    try:
        idx     = get_atom_index(row, acc_z)
        pol_arr = ensure_2d(parse_db_array(row.MWFN_MBIS_Atomic_Polarizability), 1)
        acc_polar_lookup[name] = pol_arr[idx, 0]
        acc_c6 = parse_db_array(row.MWFN_MBIS_c6)
        acc_c6_lookup[name] = acc_c6
        print(f"  {name:<8}: acc_polar = {acc_polar_lookup[name]:.6f}")
    except Exception as e:
        print(f"  WARNING - {name}: {e}")

missing = [k for k in HALOGEN_DBS if k not in acc_polar_lookup]
if missing:
    raise RuntimeError(f"Could not load acceptor polarizability for: {missing}")

# =============================================================================
# Step 2: Load interaction energies from all 4 halogen databases
# =============================================================================

print("\n" + "=" * 60)
print("Step 2 - Interaction energies")
print("=" * 60)

interaction_lookup = {}  # {acceptor_label: {mol_name: energy_kcal}}

for acceptor, (db_path, _) in HALOGEN_DBS.items():
    db   = connect(db_path)
    lkup = {}
    for row in db.select():
        mol_name = row.key_value_pairs.get("Name")
        if mol_name and "Interaction_E" in row.key_value_pairs:
            energy_kcal = row.Interaction_E * 627.5095
            lkup[mol_name] = energy_kcal  # ln(|E|)  # Eh -> kcal/mol
    interaction_lookup[acceptor] = lkup
    print(f"  {acceptor:<8} ({db_path}): {len(lkup)} molecules")

# =============================================================================
# Step 3: Extract iodine donor descriptors from mol.db
# =============================================================================

print("\n" + "=" * 60)
print(f"Step 3 - Iodine descriptors from {FEAT_DB_PATH}")
print("=" * 60)

feat_db       = connect(FEAT_DB_PATH)
donor_features = {}  # {mol_name: dict}

for row in feat_db.select():
    name = row.key_value_pairs.get("Name")
    if not name:
        continue
    try:
        i = get_iodine_index(row)

        charges  = parse_db_array(row.MWFN_MBIS_Atomic_Charges)
        c6_arr   = parse_db_array(row.MWFN_MBIS_c6)
        dip_arr  = ensure_2d(parse_db_array(row.MWFN_MBIS_Atom_Dipole), 3)
        r2_arr   = ensure_2d(parse_db_array(row.MWFN_MBIS_r2), 3)

        scaffold    = row.key_value_pairs.get("Aromtic_Scaffold", "unknown")
        ring_charge = SCAFFOLD_FACTORS.get(scaffold, 0.0)

        donor_features[name] = {
            'monopole'  : charges[i],
            'c6'        : c6_arr[i],
            'dipole_0'  : dip_arr[i, 2],
            'r2'        : r2_arr[i,0] + r2_arr[i,1] + r2_arr[i,2],
            'ring_charge': ring_charge,
            'scaffold'  : scaffold,
        }
    except Exception as e:
        print(f"  SKIP '{name}' (id={row.id}): {e}")

print(f"  {len(donor_features)} donor molecules loaded")

# =============================================================================
# Step 4: Build combined dataset
# =============================================================================

FEATURE_COLS = ['monopole', 'c6', 'dipole_0', 'r2', 'ring_charge', 'acc_polar']

print("\n" + "=" * 60)
print("Step 4 - Building combined dataset")
print("=" * 60)

records = []
for acceptor in HALOGEN_DBS:
    lkup      = interaction_lookup[acceptor]
    acc_polar = acc_polar_lookup[acceptor]
    n = 0
    for mol_name, feats in donor_features.items():
        if mol_name not in lkup:
            continue
        records.append({
            'acceptor'  : acceptor,
            'name'      : mol_name,
            'scaffold'  : feats['scaffold'],
            'monopole'  : feats['monopole'],
            'c6'        : feats['c6'],
            'dipole_0'  : feats['dipole_0'],
            'r2'        : feats['r2'],
            'ring_charge': feats['ring_charge'],
            'acc_polar' : acc_polar,
            'acc_c6'    : acc_c6,
            'target'    : lkup[mol_name],
        })
        n += 1
    print(f"  {acceptor:<8}: {n} molecule-acceptor pairs")

df_all = pd.DataFrame(records)
print(f"\n  Total rows: {len(df_all)}")
print("\n  Acceptor counts in combined dataset:")
print(df_all.groupby('acceptor').size().rename('n_rows').to_string())

# Diagnostics
print(f"\n  TEST_ACCEPTOR = '{TEST_ACCEPTOR}'")
print(f"  Rows matching test acceptor: {(df_all['acceptor'] == TEST_ACCEPTOR).sum()}")

br_names      = set(interaction_lookup.get(TEST_ACCEPTOR, {}).keys())
feat_names    = set(donor_features.keys())
missing_in_feat = br_names - feat_names
if missing_in_feat:
    print(f"\n  WARNING: {len(missing_in_feat)} '{TEST_ACCEPTOR}' molecule(s) "
          f"not found in mol.db:")
    for n in sorted(missing_in_feat)[:10]:
        print(f"    '{n}'")

# =============================================================================
# Step 5: Train / test split — leave-one-acceptor-out
# =============================================================================

print("\n" + "=" * 60)
print("Step 5 - Train / test split")
print("=" * 60)

df_train = df_all[df_all['acceptor'] != TEST_ACCEPTOR].reset_index(drop=True)
df_test  = df_all[df_all['acceptor'] == TEST_ACCEPTOR].reset_index(drop=True)

if len(df_test) == 0:
    raise RuntimeError(
        f"df_test is empty for TEST_ACCEPTOR='{TEST_ACCEPTOR}'.\n"
        f"Acceptors present: {sorted(df_all['acceptor'].unique())}"
    )

X_train = df_train[FEATURE_COLS].values
y_train = df_train['target'].values
X_test  = df_test[FEATURE_COLS].values
y_test  = df_test['target'].values

train_acceptors = sorted(df_train['acceptor'].unique())
print(f"  Train : {train_acceptors} ({len(df_train)} rows)")
print(f"  Test  : {TEST_ACCEPTOR} ({len(df_test)} rows — UNSEEN)")

# =============================================================================
# Step 6: StandardScaler (fit on train only)
# =============================================================================

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

# =============================================================================
# Step 7: Bayesian Optimisation of GB hyperparameters
#   • bayes_opt with Expected Improvement
#   • 10 LHS init points + 10 BO iterations
#   • Objective: mean RMSE, Repeated 5-Fold × 10 repeats
# =============================================================================

print("\n" + "=" * 60)
print("Step 7 - Bayesian Optimisation")
print("=" * 60)

PBOUNDS = {
    "n_estimators"             : (10,   100),
    "learning_rate"            : (0.01, 0.3),
    "max_depth"                : (2,    6),
    "min_samples_split"        : (2,    10),
    "min_samples_leaf"         : (5,    20),
    "subsample"                : (0.7,  1.0),
    "max_features"             : (0.25, 1.0),
    "validation_fraction"      : (0.1,  0.3),
    "min_weight_fraction_leaf" : (0.0,  0.05),
    "ccp_alpha"                : (0.0,  0.01),
}

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)

def gb_cv_rmse(n_estimators, learning_rate, max_depth, min_samples_split,
               min_samples_leaf, subsample, max_features, validation_fraction,
               min_weight_fraction_leaf, ccp_alpha):
    model = GradientBoostingRegressor(
        n_estimators             = int(round(n_estimators)),
        learning_rate            = learning_rate,
        max_depth                = int(round(max_depth)),
        min_samples_split        = int(round(min_samples_split)),
        min_samples_leaf         = int(round(min_samples_leaf)),
        subsample                = subsample,
        max_features             = max_features,
        validation_fraction      = validation_fraction,
        min_weight_fraction_leaf = min_weight_fraction_leaf,
        ccp_alpha                = ccp_alpha,
        random_state             = SEED,
    )
    scores = cross_val_score(
        model, X_tr_sc, y_train,
        cv=rkf, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    return scores.mean()  # negative; BO maximises → minimises RMSE

N_INIT = 10
N_ITER = 10

sampler  = qmc.LatinHypercube(d=len(PBOUNDS), seed=SEED)
lhs_unit = sampler.random(N_INIT)
lb       = np.array([v[0] for v in PBOUNDS.values()])
ub       = np.array([v[1] for v in PBOUNDS.values()])
lhs_pts  = qmc.scale(lhs_unit, lb, ub)

init_points_dict = [dict(zip(PBOUNDS.keys(), lhs_pts[i])) for i in range(N_INIT)]

optimizer = BayesianOptimization(
    f=gb_cv_rmse, pbounds=PBOUNDS, random_state=SEED, verbose=2,
)
for pt in init_points_dict:
    optimizer.probe(params=pt, lazy=True)
optimizer.maximize(init_points=0, n_iter=N_ITER)

best_params_raw = optimizer.max["params"]
best_rmse_cv    = -optimizer.max["target"]

print(f"\n  Best CV RMSE (BO): {best_rmse_cv:.4f} kcal/mol")

best_params = {
    "n_estimators"             : int(round(best_params_raw["n_estimators"])),
    "learning_rate"            : best_params_raw["learning_rate"],
    "max_depth"                : int(round(best_params_raw["max_depth"])),
    "min_samples_split"        : int(round(best_params_raw["min_samples_split"])),
    "min_samples_leaf"         : int(round(best_params_raw["min_samples_leaf"])),
    "subsample"                : best_params_raw["subsample"],
    "max_features"             : best_params_raw["max_features"],
    "validation_fraction"      : best_params_raw["validation_fraction"],
    "min_weight_fraction_leaf" : best_params_raw["min_weight_fraction_leaf"],
    "ccp_alpha"                : best_params_raw["ccp_alpha"],
    "random_state"             : SEED,
}
print("\n  Best Hyperparameters:")
for k, v in best_params.items():
    print(f"    {k:>30}: {v}")

# =============================================================================
# Step 8: Final model on full training set
# =============================================================================

print("\n" + "=" * 60)
print("Step 8 - Final model")
print("=" * 60)

final_model = GradientBoostingRegressor(**best_params)
final_model.fit(X_tr_sc, y_train)

y_pred_train = final_model.predict(X_tr_sc)
y_pred_test  = final_model.predict(X_te_sc)

r2_train   = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_test    = r2_score(y_test, y_pred_test)
rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n  Training ({', '.join(train_acceptors)})")
print(f"    R²   : {r2_train:.4f}")
print(f"    RMSE : {rmse_train:.4f} kcal/mol")
print(f"\n  Held-out test ({TEST_ACCEPTOR})")
print(f"    R²   : {r2_test:.4f}")
print(f"    RMSE : {rmse_test:.4f} kcal/mol")

# =============================================================================
# Step 9: Final CV on training set with best hyperparameters
# =============================================================================

cv_scores_rmse = -cross_val_score(
    GradientBoostingRegressor(**best_params), X_tr_sc, y_train,
    cv=rkf, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
cv_scores_r2 = cross_val_score(
    GradientBoostingRegressor(**best_params), X_tr_sc, y_train,
    cv=rkf, scoring="r2", n_jobs=-1,
)

print("\n  Final CV (Repeated 5-Fold × 10) with Best Params:")
print(f"    RMSE : {cv_scores_rmse.mean():.4f} ± {cv_scores_rmse.std():.4f} kcal/mol")
print(f"    R²   : {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")

# =============================================================================
# Step 10: Feature importances
# =============================================================================

importances = final_model.feature_importances_
print("\n  Feature Importances:")
for fname, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    print(f"    {fname:>20}: {imp:.4f}")

# =============================================================================
# Plotting
# =============================================================================

scaffolds_test  = df_test['scaffold'].to_numpy()
scaffolds_train = df_train['scaffold'].to_numpy()
all_scaffolds   = sorted(set(df_all['scaffold'].unique()))

palette = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22',
]
color_map = {s: palette[i % len(palette)] for i, s in enumerate(all_scaffolds)}

# ── Parity plot — test set coloured by scaffold (matches adj-model style) ─────
fig, ax = plt.subplots(figsize=(7, 7))

for scaffold in all_scaffolds:
    mask = scaffolds_test == scaffold
    if not mask.any():
        continue
    ax.scatter(
        y_test[mask], y_pred_test[mask],
        facecolors=color_map[scaffold], edgecolors='black',
        linewidths=0.5, s=55, zorder=3, label=scaffold,
    )
for scaffold in all_scaffolds:
    mask = scaffolds_train == scaffold
    if mask.any():
        ax.scatter(y_train[mask], y_pred_train[mask],
                   facecolors='grey',
                   linewidths=1.2, s=40, zorder=2, label=f'Training Set')


all_vals = np.concatenate([y_test, y_pred_test])
pad  = 0.07 * (all_vals.max() - all_vals.min())
lims = [all_vals.min() - pad, all_vals.max() + pad]
ax.plot(lims, lims, 'k--', lw=1.0, zorder=1)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
ax.set_xlabel('DFT E_int (kcal/mol)', fontsize=13)
ax.set_ylabel('Predicted E_int (kcal/mol)', fontsize=13)
metric_handles = [
    Line2D([], [], color='none', label=f'Training R² = {r2_train:.3f} RMSE = {rmse_train:.3f} kcal/mol'),
    Line2D([], [], color='none', label=f'Test R² = {r2_test:.3f} RMSE = {rmse_test:.3f} kcal/mol'),
    Line2D([], [], color='black', ls='--', lw=1.0, label='Perfect fit'),
]
sh, sl = ax.get_legend_handles_labels()
ax.legend(handles=metric_handles + sh,
          labels=[h.get_label() for h in metric_handles] + sl,
          fontsize=11, framealpha=1, loc='upper left')
fig.tight_layout()
save_fig(fig, f"gb_parity_{TEST_ACCEPTOR}")

# ── Residual plot ─────────────────────────────────────────────────────────────
resid_train = y_train - y_pred_train
resid_test  = y_test  - y_pred_test

fig, ax = plt.subplots(figsize=(7, 6))
for scaffold in all_scaffolds:
    mask = scaffolds_train == scaffold
    if mask.any():
        ax.scatter(y_pred_train[mask], resid_train[mask],
                   facecolors='none', edgecolors=color_map[scaffold],
                   linewidths=1.2, s=40, zorder=2, label=f'{scaffold} (train)')
for scaffold in all_scaffolds:
    mask = scaffolds_test == scaffold
    if mask.any():
        ax.scatter(y_pred_test[mask], resid_test[mask],
                   facecolors=color_map[scaffold], edgecolors='black',
                   linewidths=0.5, s=55, zorder=3, marker='D',
                   label=f'{scaffold} (test)')
ax.axhline(0, color='black', ls='--', lw=1.0)
all_pred = np.concatenate([y_pred_train, y_pred_test])
all_res  = np.concatenate([resid_train,  resid_test])
ax.set_xlim(all_pred.min() - 0.05*(all_pred.max()-all_pred.min()),
            all_pred.max() + 0.05*(all_pred.max()-all_pred.min()))
ax.set_ylim(all_res.min()  - 0.05*(all_res.max()-all_res.min()),
            all_res.max()  + 0.05*(all_res.max()-all_res.min()))
ax.set_xlabel('Predicted Interaction Energy (kcal/mol)', fontsize=11)
ax.set_ylabel('Residual (DFT − Predicted) (kcal/mol)', fontsize=11)
ax.set_title(f'Residual Plot — trained on {", ".join(train_acceptors)}, '
             f'tested on {TEST_ACCEPTOR}', fontsize=11)
sh, sl = ax.get_legend_handles_labels()
ax.legend(handles=sh, labels=sl, fontsize=7, framealpha=0.9, loc='upper left')
fig.tight_layout()
save_fig(fig, f"gb_residual_{TEST_ACCEPTOR}")

# ── KDE of residuals ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (resid, split_label) in zip(
    axes,
    [(resid_train, f"Training Residuals ({', '.join(train_acceptors)})"),
     (resid_test,  f"Test Residuals ({TEST_ACCEPTOR} — unseen)")]
):
    kde = gaussian_kde(resid, bw_method="scott")
    xs  = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 400)
    ax.plot(xs, kde(xs), color='#1f77b4', lw=2.0)
    ax.axvline(resid.mean(), color='#1f77b4', ls=':', lw=1.5,
               label=f'Mean = {resid.mean():.3f} kcal/mol')
    ax.axvline(0, color='black', ls='--', lw=1.0, label='Zero residual')
    ax.set_xlabel('Residual (kcal/mol)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(split_label, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)
fig.suptitle('GB Model — KDE of Residuals', fontsize=12, y=1.01)
fig.tight_layout()
save_fig(fig, f"gb_kde_residuals_{TEST_ACCEPTOR}")

# ── Feature importance bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
sorted_pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1])
fnames_sorted, imps_sorted = zip(*sorted_pairs)
bars = ax.barh(fnames_sorted, imps_sorted, color='#1f77b4', edgecolor='black')
ax.set_xlabel('Feature Importance (MDI)', fontsize=11)
ax.set_title('GB Model — Feature Importances', fontsize=12)
for bar, val in zip(bars, imps_sorted):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)
fig.tight_layout()
save_fig(fig, "gb_feature_importance")

# ── BO convergence trace ──────────────────────────────────────────────────────
bo_targets = [-r["target"] for r in optimizer.res]
bo_best    = np.minimum.accumulate(bo_targets)
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(range(1, len(bo_targets)+1), bo_targets,
           color='grey', s=30, zorder=2, label='Probe RMSE')
ax.plot(range(1, len(bo_best)+1), bo_best,
        color='#d62728', lw=2.0, zorder=3, label='Best so far')
ax.axvline(N_INIT + 0.5, color='black', ls=':', lw=1.0,
           label=f'LHS init / BO boundary (n={N_INIT})')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('CV RMSE (kcal/mol)', fontsize=11)
ax.set_title('Bayesian Optimisation Convergence', fontsize=12)
ax.legend(fontsize=9, framealpha=0.9)
fig.tight_layout()
save_fig(fig, "gb_bo_convergence")

# ── Learning curve ────────────────────────────────────────────────────────────
train_sizes, train_scores, cv_scores = learning_curve(
    GradientBoostingRegressor(**best_params),
    X_tr_sc, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
train_rmse = -train_scores.mean(axis=1)
cv_rmse    = -cv_scores.mean(axis=1)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(train_sizes, train_rmse, 'o-', color='#1f77b4', label='Training RMSE')
ax.plot(train_sizes, cv_rmse,   'o-', color='#d62728', label='CV RMSE')
ax.fill_between(train_sizes, train_rmse - train_scores.std(axis=1),
                train_rmse + train_scores.std(axis=1), alpha=0.15, color='#1f77b4')
ax.fill_between(train_sizes, cv_rmse - cv_scores.std(axis=1),
                cv_rmse + cv_scores.std(axis=1), alpha=0.15, color='#d62728')
ax.set_xlabel('Training set size', fontsize=11)
ax.set_ylabel('RMSE (kcal/mol)', fontsize=11)
ax.set_title(f'Learning Curve ({", ".join(train_acceptors)} training data)', fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "gb_learning_curve")

# =============================================================================
# Prediction helper
# =============================================================================

def predict_interaction_energy(monopole, c6, dipole_0, r2,
                                scaffold_name, acceptor_label):
    """
    Predict interaction energy (kcal/mol) for a single donor-acceptor pair.

    Parameters
    ----------
    monopole, c6, dipole_0, r2 : float
        Iodine MBIS descriptors from the donor molecule.
    scaffold_name : str
        Donor aromatic scaffold (key in SCAFFOLD_FACTORS; 0.0 e if unknown).
    acceptor_label : str
        One of: 'nan-I', 'nan-F', 'nan-Cl', 'nan-Br'
    """
    ring_charge = SCAFFOLD_FACTORS.get(scaffold_name, 0.0)
    acc_polar   = acc_polar_lookup[acceptor_label]
    acc_c6 = acc_c6_lookup[acceptor_label]
    X = scaler.transform([[monopole, c6, dipole_0, r2, ring_charge, acc_polar,acc_c6]])
    return float(final_model.predict(X)[0])
