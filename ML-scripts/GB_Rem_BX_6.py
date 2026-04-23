"""
gb-model.py

Trains a Gradient Boosting Regressor on interaction energies using the same
methodology as ROBERT:
  - Features : I_monopole, I_dipole_0, I_C6, I_r2, ring_charge,
               acc_monopole, acc_dipole  (ROBERT selection + iodine descriptors + scaffold)
  - Hyperopt : Bayesian Optimisation (Expected Improvement, bayes_opt library)
               with Latin Hypercube Sampling for initialisation
  - CV       : Repeated 5-Fold × 10 repeats, scored by RMSE
  - Split    : BX-6 scaffold held out as test set (leave-one-scaffold-out)
               Change TEST_SCAFFOLD below to hold out a different scaffold
  - Scaling  : StandardScaler fit on train only
  - Seed     : 0  (ROBERT default)

Feature extraction mirrors extract-dataset.py exactly.
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
from sklearn.pipeline import Pipeline

from bayes_opt import BayesianOptimization
from scipy.stats import qmc                    # Latin Hypercube Sampling

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

SEED = 0

# ══════════════════════════════════════════════════════════════════════════════
# Helpers  (identical to extract-dataset.py)
# ══════════════════════════════════════════════════════════════════════════════

def parse_db_array(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    elif isinstance(raw, np.ndarray):
        raw = raw.tolist()
    return np.array(raw)

def get_iodine_index(row) -> int:
    for idx, z in enumerate(row.numbers):
        if z == 53:
            return idx
    raise ValueError(
        f"No iodine in molecule '{row.key_value_pairs.get('Name', row.id)}'"
    )

def get_first_atom_index(row, symbol: str) -> int:
    from ase.data import chemical_symbols
    target_z = chemical_symbols.index(symbol)
    for idx, z in enumerate(row.numbers):
        if z == target_z:
            return idx
    raise ValueError(
        f"No '{symbol}' atom in '{row.key_value_pairs.get('Name', row.id)}'"
    )

def get_all_atom_indices(row, symbol: str) -> list:
    from ase.data import chemical_symbols
    target_z = chemical_symbols.index(symbol)
    return [idx for idx, z in enumerate(row.numbers) if z == target_z]

# ══════════════════════════════════════════════════════════════════════════════
# Target DB config  (identical to extract-dataset.py)
# ══════════════════════════════════════════════════════════════════════════════

SCAFFOLD_FACTORS = {
    "Benzene"    : -0.666233,
    "Pyrimidine" : -0.326117,
    "BX-5"       : -0.648058,
    "BX-6"       : -0.648042,
    "BZ-5"       : -0.642792,
    "BZ-6"       : -0.642793,
}


TARGET_DB_CONFIG = {
    "I-molecules.db"       : ("I",       "I",       "I"),
    "Cl-molecules.db"      : ("Cl",      "Cl",      "Cl"),
    "Br-molecules.db"      : ("Br",      "Br",      "Br"),
    "F-molecules.db"       : ("F",       "F",       "F"),
    "Ketone-molecules.db"   : ("Ketone",  "Ketone",  "O"),
    "NH3-molecules.db"      : ("NH3",     "NH3",     "N"),
    "NF3-molecules.db"      : ("NF3",     "NF3",     "N"),
    "PH3-molecules.db"      : ("PH3",     "PH3",     "P"),
    "Benzene-molecules.db"  : ("Benzene", "Benzene", "benzene_C"),
}

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# Change TEST_SCAFFOLD to hold out a different scaffold for blind prediction.
# Valid options: "Benzene", "Pyrimidine", "BX-5", "BX-6", "BZ-5", "BZ-6"
# ══════════════════════════════════════════════════════════════════════════════

TEST_SCAFFOLD = ["BX-6"]

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — read mol.db: iodine descriptors + scaffold
# ══════════════════════════════════════════════════════════════════════════════

feat_db     = connect("mol.db")
feat_lookup = {}

for row in feat_db.select():
    mol_name = row.key_value_pairs.get("Name")
    if mol_name is None:
        continue
    try:
        i_idx = get_iodine_index(row)
    except ValueError as e:
        print(f"  SKIP mol.db '{mol_name}': {e}")
        continue

    monopole    = float(parse_db_array(row.MWFN_MBIS_Atomic_Charges)[i_idx])
    dipole_0    = float(parse_db_array(row.MWFN_MBIS_Atom_Dipole)[i_idx, 2])
    c6          = float(parse_db_array(row.MWFN_MBIS_c6)[i_idx])
    R2          = parse_db_array(row.MWFN_MBIS_r2)
    r2          = float(R2[i_idx, 0] + R2[i_idx, 1] + R2[i_idx, 2])
    scaffold    = row.key_value_pairs.get("Aromtic_Scaffold", "unknown")
    ring_charge = SCAFFOLD_FACTORS.get(scaffold, 0.0)

    feat_lookup[mol_name] = dict(
        I_monopole  = monopole,
        I_dipole_0  = dipole_0,
        I_C6        = c6,
        I_r2        = r2,
        scaffold    = scaffold,
        ring_charge = ring_charge,
    )

print(f"mol.db: {len(feat_lookup)} molecules parsed")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — read acceptor.db: acc_monopole, acc_dipole
# ══════════════════════════════════════════════════════════════════════════════

acceptor_db     = connect("acceptor.db")
acceptor_lookup = {}
for row in acceptor_db.select():
    acc_name = row.key_value_pairs.get("Name")
    if acc_name is not None:
        acceptor_lookup[acc_name] = row

def extract_acceptor_features(acceptor_name: str, atom_rule: str) -> dict:
    if acceptor_name not in acceptor_lookup:
        raise KeyError(f"'{acceptor_name}' not found in acceptor.db")
    row          = acceptor_lookup[acceptor_name]
    monopole_arr = parse_db_array(row.MWFN_MBIS_Atomic_Charges)

    if atom_rule == "benzene_C":
        acc_monopole = float(np.sum(monopole_arr))
    else:
        idx          = get_first_atom_index(row, atom_rule)
        acc_monopole = float(monopole_arr[idx])

    # Molecular dipole magnitude — all acceptors
    dipole_vec = parse_db_array(row.ORCA_Mol_Dipole)
    acc_dipole = float(np.linalg.norm(dipole_vec))

    return dict(acc_monopole=acc_monopole, acc_dipole=acc_dipole)

acceptor_cache = {}
for db_filename, (interactant, acc_name, atom_rule) in TARGET_DB_CONFIG.items():
    try:
        acceptor_cache[acc_name] = extract_acceptor_features(acc_name, atom_rule)
    except Exception as e:
        print(f"WARNING: acceptor features for '{acc_name}' failed: {e}")

print(f"acceptor.db: {len(acceptor_cache)} acceptors parsed")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — assemble dataset
# ══════════════════════════════════════════════════════════════════════════════

records = []

for db_filename, (interactant, acc_name, atom_rule) in TARGET_DB_CONFIG.items():
    try:
        tdb = connect(db_filename)
    except Exception as e:
        print(f"WARNING: Could not open '{db_filename}': {e}")
        continue
    if acc_name not in acceptor_cache:
        continue

    acc_feats = acceptor_cache[acc_name]
    n_added   = 0

    for row in tdb.select():
        mol_name = row.key_value_pairs.get("Name")
        if mol_name is None:
            continue
        if "Interaction_E" not in row.key_value_pairs:
            continue
        if mol_name not in feat_lookup:
            continue

        mol_feats     = feat_lookup[mol_name]
        interaction_E = row.Interaction_E * 627.5095   # Hartree → kcal/mol
        scaffold      = mol_feats["scaffold"]

        records.append(dict(
            name          = f"{interactant}-{scaffold}-{mol_name}",
            target        = interactant,
            scaffold      = scaffold,
            interaction_E = interaction_E,
            I_monopole    = mol_feats["I_monopole"],
            I_dipole_0    = mol_feats["I_dipole_0"],
            I_C6          = mol_feats["I_C6"],
            I_r2          = mol_feats["I_r2"],
            ring_charge   = mol_feats["ring_charge"],
            acc_monopole  = acc_feats["acc_monopole"],
            acc_dipole    = acc_feats["acc_dipole"],
        ))
        n_added += 1

    print(f"  {db_filename}: {n_added} entries")

full_df = pd.DataFrame(records)
print(f"\nTotal dataset: {len(full_df)} entries")

FEATURES = [
    "I_monopole",
    "I_dipole_0",
    "I_C6",
    "I_r2",
    "ring_charge",
    "acc_monopole",
    "acc_dipole",
]
X_all       = full_df[FEATURES].values
y_all       = full_df["interaction_E"].values

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Leave-one-scaffold-out split: TEST_SCAFFOLD → test, rest → train
# ══════════════════════════════════════════════════════════════════════════════

scaffold_mask = full_df["scaffold"].isin(TEST_SCAFFOLD).values
idx_test      = np.where( scaffold_mask)[0]
idx_train     = np.where(~scaffold_mask)[0]

X_train, y_train = X_all[idx_train], y_all[idx_train]
X_test,  y_test  = X_all[idx_test],  y_all[idx_test]

train_targets = sorted(full_df.iloc[idx_train]["target"].unique())
print(f"Train: {len(y_train)} entries (scaffolds ≠ {TEST_SCAFFOLD!r}, acceptors: {train_targets})")
print(f"Test : {len(y_test)}  entries (scaffold  = {TEST_SCAFFOLD!r}, all acceptors)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — StandardScaler (fit on train only, as ROBERT does)
# ══════════════════════════════════════════════════════════════════════════════

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Bayesian Optimisation (ROBERT methodology)
#   • bayes_opt library with Expected Improvement (xi=0.05)
#   • 10 init points seeded by Latin Hypercube Sampling
#   • 10 BO iterations
#   • Objective: mean RMSE across Repeated 5-Fold × 10 repeats
# ══════════════════════════════════════════════════════════════════════════════

# BO search bounds (from ROBERT GB_params.yaml / BO_hyperparams)
PBOUNDS = {
    "n_estimators"              : (10,   100),
    "learning_rate"             : (0.01, 0.3),
    "max_depth"                 : (2,    6),
    "min_samples_split"         : (2,    10),
    "min_samples_leaf"          : (5,    20),
    "subsample"                 : (0.7,  1.0),
    "max_features"              : (0.25, 1.0),
    "validation_fraction"       : (0.1,  0.3),
    "min_weight_fraction_leaf"  : (0.0,  0.05),
    "ccp_alpha"                 : (0.0,  0.01),
}

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)

def gb_cv_rmse(n_estimators, learning_rate, max_depth, min_samples_split,
               min_samples_leaf, subsample, max_features, validation_fraction,
               min_weight_fraction_leaf, ccp_alpha):
    """Objective function for BO: returns negative mean RMSE (BO maximises)."""
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
    return scores.mean()   # already negative; BO maximises → minimises RMSE

# Latin Hypercube Sampling for initial BO points
N_INIT = 10
N_ITER = 10

sampler   = qmc.LatinHypercube(d=len(PBOUNDS), seed=SEED)
lhs_unit  = sampler.random(N_INIT)                         # shape (N_INIT, d)
lb        = np.array([v[0] for v in PBOUNDS.values()])
ub        = np.array([v[1] for v in PBOUNDS.values()])
lhs_pts   = qmc.scale(lhs_unit, lb, ub)                   # scale to bounds

init_points_dict = [
    dict(zip(PBOUNDS.keys(), lhs_pts[i])) for i in range(N_INIT)
]

optimizer = BayesianOptimization(
    f            = gb_cv_rmse,
    pbounds      = PBOUNDS,
    random_state = SEED,
    verbose      = 2,
)

# Register LHS points as initial probes
for pt in init_points_dict:
    optimizer.probe(params=pt, lazy=True)

# Run BO
optimizer.maximize(init_points=0, n_iter=N_ITER)

best_params_raw = optimizer.max["params"]
best_rmse_cv    = -optimizer.max["target"]

print(f"\n── Best CV RMSE (BO): {best_rmse_cv:.4f} kcal/mol ──────────────────")

best_params = {
    "n_estimators"            : int(round(best_params_raw["n_estimators"])),
    "learning_rate"           : best_params_raw["learning_rate"],
    "max_depth"               : int(round(best_params_raw["max_depth"])),
    "min_samples_split"       : int(round(best_params_raw["min_samples_split"])),
    "min_samples_leaf"        : int(round(best_params_raw["min_samples_leaf"])),
    "subsample"               : best_params_raw["subsample"],
    "max_features"            : best_params_raw["max_features"],
    "validation_fraction"     : best_params_raw["validation_fraction"],
    "min_weight_fraction_leaf": best_params_raw["min_weight_fraction_leaf"],
    "ccp_alpha"               : best_params_raw["ccp_alpha"],
    "random_state"            : SEED,
}

print("\n── Best Hyperparameters ──────────────────────────────────────────────")
for k, v in best_params.items():
    print(f"  {k:>30}: {v}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Final model trained on full training set
# ══════════════════════════════════════════════════════════════════════════════

final_model = GradientBoostingRegressor(**best_params)
final_model.fit(X_tr_sc, y_train)

y_pred_train = final_model.predict(X_tr_sc)
y_pred_test  = final_model.predict(X_te_sc)

r2_train   = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_test    = r2_score(y_test,  y_pred_test)
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))

print(f"\n── Training Set Performance (scaffolds ≠ {TEST_SCAFFOLD!r}) ─────────────")
print(f"  R²   : {r2_train:.4f}")
print(f"  RMSE : {rmse_train:.4f} kcal/mol")

print(f"\n── Held-Out Test Set Performance ({TEST_SCAFFOLD!r} scaffold) ───────────────")
print(f"  R²   : {r2_test:.4f}")
print(f"  RMSE : {rmse_test:.4f} kcal/mol")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Final CV on full training set with best hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

cv_scores_rmse = -cross_val_score(
    GradientBoostingRegressor(**best_params),
    X_tr_sc, y_train,
    cv=rkf, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
cv_scores_r2 = cross_val_score(
    GradientBoostingRegressor(**best_params),
    X_tr_sc, y_train,
    cv=rkf, scoring="r2", n_jobs=-1,
)

print("\n── Final CV (Repeated 5-Fold × 10) with Best Params ──────────────────")
print(f"  RMSE : {cv_scores_rmse.mean():.4f} ± {cv_scores_rmse.std():.4f} kcal/mol")
print(f"  R²   : {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Feature importances
# ══════════════════════════════════════════════════════════════════════════════

importances = final_model.feature_importances_
print("\n── Feature Importances ───────────────────────────────────────────────")
for fname, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    print(f"  {fname:>20}: {imp:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

targets_train = full_df.iloc[idx_train]["target"].to_numpy()
targets_test  = full_df.iloc[idx_test]["target"].to_numpy()
all_targets   = sorted(full_df["target"].unique())

palette = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22',
]
color_map = {t: palette[i % len(palette)] for i, t in enumerate(all_targets)}

def save_fig(fig, stem):
    for ext in ("pdf", "svg", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"Saved → {path}")
    plt.close(fig)

# ── Parity plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

for tgt in all_targets:
    mask = targets_train == tgt
    if mask.any():
        ax.scatter(y_train[mask], y_pred_train[mask],
                   facecolors='grey', edgecolors='none',
                   linewidths=1.2, s=50, zorder=2, label=f'Training Set')

for tgt in all_targets:
    mask = targets_test == tgt
    if mask.any():
        ax.scatter(y_test[mask], y_pred_test[mask],
                   facecolors=color_map[tgt], edgecolors='black',
                   linewidths=0.5, s=50, zorder=3, label=f'{tgt} (test)')

all_vals = np.concatenate([y_train, y_test, y_pred_train, y_pred_test])
pad  = 0.05 * (all_vals.max() - all_vals.min())
lims = [all_vals.min() - pad, all_vals.max() + pad]
ax.plot(lims, lims, 'k--', lw=1.0, zorder=1)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
ax.set_xlabel(r'$E_{int}^{DFT} \mathrm{/(kcal/mol)}$', fontsize=13)
ax.set_ylabel(r'$E_{int}^{Pred} \mathrm{/(kcal/mol)}$', fontsize=13)

mh = [
    Line2D([], [], color='none',
           label=f'Test   R²={r2_test:.3f}  RMSE={rmse_test:.3f} kcal/mol'),
    Line2D([], [], color='none',
           label=f'Train  R²={r2_train:.3f}  RMSE={rmse_train:.3f} kcal/mol'),
    Line2D([], [], color='black', ls='--', lw=1.0, label='Perfect fit'),
]
sh, sl = ax.get_legend_handles_labels()
ax.legend(handles=mh + sh,
          labels=[h.get_label() for h in mh] + sl,
          fontsize=11, framealpha=0.9, loc='upper left')
fig.tight_layout()
save_fig(fig, "gb_parity")

# ── Residual plot ─────────────────────────────────────────────────────────────
resid_train = y_train - y_pred_train
resid_test  = y_test  - y_pred_test

fig, ax = plt.subplots(figsize=(7, 6))

for tgt in all_targets:
    mask = targets_train == tgt
    if mask.any():
        ax.scatter(y_pred_train[mask], resid_train[mask],
                   facecolors='none', edgecolors=color_map[tgt],
                   linewidths=1.2, s=50, zorder=2, label=f'{tgt} (train)')
for tgt in all_targets:
    mask = targets_test == tgt
    if mask.any():
        ax.scatter(y_pred_test[mask], resid_test[mask],
                   facecolors=color_map[tgt], edgecolors='black',
                   linewidths=0.5, s=50, zorder=3, label=f'{tgt} (test)')

ax.axhline(0, color='black', ls='--', lw=1.0)
all_pred = np.concatenate([y_pred_train, y_pred_test])
all_res  = np.concatenate([resid_train,  resid_test])
xpad = 0.05 * (all_pred.max() - all_pred.min())
ypad = 0.05 * (all_res.max()  - all_res.min())
ax.set_xlim(all_pred.min() - xpad, all_pred.max() + xpad)
ax.set_ylim(all_res.min()  - ypad, all_res.max()  + ypad)
ax.set_xlabel('Predicted Interaction Energy (kcal/mol)', fontsize=11)
ax.set_ylabel('Residual (DFT − Predicted) (kcal/mol)',   fontsize=11)
ax.set_title(f'GB Model — Residual Plot\n(test: {TEST_SCAFFOLD} scaffold, train: all others)', fontsize=12)
sh, sl = ax.get_legend_handles_labels()
ax.legend(handles=sh, labels=sl, fontsize=7, framealpha=0.9, loc='upper left')
fig.tight_layout()
save_fig(fig, "gb_residual")

# ── KDE of residuals ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, (resid, split_label) in zip(
    axes,
    [(resid_train, f"Training Residuals (scaffolds ≠ {TEST_SCAFFOLD!r})"),
     (resid_test,  f"Test Residuals ({TEST_SCAFFOLD!r} scaffold — unseen)")]
):
    kde = gaussian_kde(resid, bw_method="scott")
    xs  = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 400)
    ax.plot(xs, kde(xs), color='#1f77b4', lw=2.0, label='GB model')
    ax.axvline(resid.mean(), color='#1f77b4', ls=':', lw=1.5,
               label=f'Mean = {resid.mean():.3f} kcal/mol')
    ax.axvline(0, color='black', ls='--', lw=1.0, label='Zero residual')
    ax.set_xlabel('Residual (kcal/mol)', fontsize=10)
    ax.set_ylabel('Density',             fontsize=10)
    ax.set_title(split_label,            fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)

fig.suptitle('GB Model — KDE of Residuals', fontsize=12, y=1.01)
fig.tight_layout()
save_fig(fig, "gb_kde_residuals")

# ── Feature importance bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
sorted_pairs = sorted(zip(FEATURES, importances), key=lambda x: x[1])
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
bo_targets = [-r["target"] for r in optimizer.res]   # RMSE values
bo_best    = np.minimum.accumulate(bo_targets)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(range(1, len(bo_targets) + 1), bo_targets,
           color='grey', s=30, zorder=2, label='Probe RMSE')
ax.plot(range(1, len(bo_best) + 1), bo_best,
        color='#d62728', lw=2.0, zorder=3, label='Best so far')
ax.axvline(N_INIT + 0.5, color='black', ls=':', lw=1.0,
           label=f'LHS init / BO boundary (n={N_INIT})')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('CV RMSE (kcal/mol)', fontsize=11)
ax.set_title('Bayesian Optimisation Convergence', fontsize=12)
ax.legend(fontsize=9, framealpha=0.9)
fig.tight_layout()
save_fig(fig, "gb_bo_convergence")

# ── Learning Curve ────────────────────────────────────────────────────── 

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
train_std  = train_scores.std(axis=1)
cv_std     = cv_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(train_sizes, train_rmse, 'o-', color='#1f77b4', label='Training RMSE')
ax.plot(train_sizes, cv_rmse,   'o-', color='#d62728', label='CV RMSE')
ax.fill_between(train_sizes,
                train_rmse - train_std, train_rmse + train_std,
                alpha=0.15, color='#1f77b4')
ax.fill_between(train_sizes,
                cv_rmse - cv_std, cv_rmse + cv_std,
                alpha=0.15, color='#d62728')
ax.set_xlabel('Training set size', fontsize=11)
ax.set_ylabel('RMSE (kcal/mol)',   fontsize=11)
ax.set_title('Learning Curve',     fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "gb_learning_curve")

# ══════════════════════════════════════════════════════════════════════════════
# Prediction helper
# ══════════════════════════════════════════════════════════════════════════════

def predict(I_monopole: float, I_dipole_0: float, I_C6: float, I_r2: float,
            scaffold_name: str, acc_monopole: float, acc_dipole: float) -> float:
    """
    Predict interaction energy (kcal/mol) using the trained GB model.
      I_monopole    — MBIS atomic charge on iodine
      I_dipole_0    — z-component of atomic dipole on iodine
      I_C6          — C6 dispersion coefficient on iodine
      I_r2          — sum of <r²> components on iodine
      scaffold_name — aromatic scaffold name (looked up in SCAFFOLD_FACTORS)
      acc_monopole  — MBIS charge on acceptor atom (or sum for Benzene)
      acc_dipole    — ORCA molecular dipole magnitude of acceptor
    """
    ring_charge = SCAFFOLD_FACTORS.get(scaffold_name, 0.0)
    X = scaler.transform([[I_monopole, I_dipole_0, I_C6, I_r2,
                           ring_charge, acc_monopole, acc_dipole]])
    return float(final_model.predict(X)[0])
