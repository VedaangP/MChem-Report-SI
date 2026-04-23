from ase.db import connect
import numpy as np
import json
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def get_iodine_index(row) -> int:
    for idx, atomic_number in enumerate(row.numbers):
        if atomic_number == 53:
            return idx
    name = row.key_value_pairs.get("Name", f"id={row.id}")
    raise ValueError(f"No iodine atom found in molecule '{name}'")

def get_atom_index_by_symbol(row, symbol: str) -> int:
    """Return the index of the first atom matching the given chemical symbol."""
    from ase.data import chemical_symbols
    target_z = chemical_symbols.index(symbol)
    for idx, z in enumerate(row.numbers):
        if z == target_z:
            return idx
    name = row.key_value_pairs.get("Name", f"id={row.id}")
    raise ValueError(f"No '{symbol}' atom found in acceptor molecule '{name}'")

def get_all_indices_by_symbol(row, symbol: str) -> list:
    """Return indices of ALL atoms matching the given chemical symbol."""
    from ase.data import chemical_symbols
    target_z = chemical_symbols.index(symbol)
    return [idx for idx, z in enumerate(row.numbers) if z == target_z]

def parse_db_array(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    elif isinstance(raw, np.ndarray):
        raw = raw.tolist()
    return np.array(raw)

# ══════════════════════════════════════════════════════════════════════════════
# Scaffold ring partial charges (units: e)
# ══════════════════════════════════════════════════════════════════════════════

SCAFFOLD_FACTORS = {
    "Benzene"    : -0.666233,
    "Pyrimidine" : -0.326117,
    "BX-5"       : -0.648058,
    "BX-6"       : -0.648042,
    "BZ-5"       : -0.642792,
    "BZ-6"       : -0.642793,
}

# ══════════════════════════════════════════════════════════════════════════════
# Target database definitions
# Maps db filename → (acceptor name in acceptor.db, atom extraction rule)
# Rule is either a chemical symbol string, or "benzene_C" for the special case.
# ══════════════════════════════════════════════════════════════════════════════

TARGET_DB_CONFIG = {
    "ASE-databases/I-molecules.db"       : ("I",       "I"),
    "ASE-databases/Cl-molecules.db"      : ("Cl",      "Cl"),
    "ASE-databases/Br-molecules.db"      : ("Br",      "Br"),
    "ASE-databases/F-molecules.db"       : ("F",       "F"),
    "ASE-databases/Ketone-molecules.db"  : ("Ketone",  "O"),
    "ASE-databases/NH3-molecules.db"     : ("NH3",     "N"),
    "ASE-databases/NF3-molecules.db"     : ("NF3",     "N"),
    "ASE-databases/PH3-molecules.db"     : ("PH3",     "P"),
    "ASE-databases/Benzene-molecules.db" : ("Benzene", "benzene_C"),
}

# ══════════════════════════════════════════════════════════════════════════════
# Load acceptor.db once — shared by both models
# ══════════════════════════════════════════════════════════════════════════════

acceptor_db = connect("ASE-databases/acceptor.db")

acceptor_lookup = {}
for row in acceptor_db.select():
    acc_name = row.key_value_pairs.get("Name")
    if acc_name is not None:
        acceptor_lookup[acc_name] = row

def extract_acceptor_features(acceptor_name: str, atom_rule: str):
    """
    Returns (polarization, monopole, dipole) for the given acceptor.

    atom_rule:
      - chemical symbol string  → use the first atom of that element
      - "benzene_C"             → average polarization over all C atoms,
                                  use molecular monopole and dipole magnitude
    """
    if acceptor_name not in acceptor_lookup:
        raise KeyError(f"Acceptor '{acceptor_name}' not found in acceptor.db")

    row = acceptor_lookup[acceptor_name]

    polarization_arr = parse_db_array(row.MWFN_MBIS_Atomic_Polarizability)
    monopole_arr     = parse_db_array(row.MWFN_MBIS_Atomic_Charges)
    dipole_arr       = parse_db_array(row.MWFN_MBIS_Atom_Dipole)

    if atom_rule == "benzene_C":
        c_indices    = get_all_indices_by_symbol(row, "C")
        polarization = np.mean([polarization_arr[i, 0] for i in c_indices])
        monopole     = float(np.sum(monopole_arr))
        dipole_total = np.sum(dipole_arr, axis=0)
        dipole       = float(np.linalg.norm(dipole_total))
    else:
        idx          = get_atom_index_by_symbol(row, atom_rule)
        polarization = float(polarization_arr[idx, 0])
        monopole     = float(monopole_arr[idx])
        dipole_total = np.sum(dipole_arr, axis=0)
        dipole       = float(np.linalg.norm(dipole_total))


    return polarization, monopole, dipole

# ══════════════════════════════════════════════════════════════════════════════
# Read feat_db (mol.db) — extract all iodine descriptors in one pass
# ══════════════════════════════════════════════════════════════════════════════

feat_db = connect("ASE-databases/mol.db")

feat_lookup = {}
# Name → {vmax, monopole, c6, dipole_0, r2, scaffold, ring_charge}

for row in feat_db.select():
    name = row.key_value_pairs.get("Name")
    if name is None:
        continue

    i_index = get_iodine_index(row)

    # Vmax on iodine
    vmin_vmax = parse_db_array(row.MWFN_MBIS_Iodine_Vmin_Vmax)
    if vmin_vmax.ndim == 2:
        vmax = float(vmin_vmax[i_index, 1])
    else:
        vmax = float(vmin_vmax[1])

    # Full iodine MBIS descriptors
    monopole = float(parse_db_array(row.MWFN_MBIS_Atomic_Charges)[i_index])
    c6       = float(parse_db_array(row.MWFN_MBIS_c6)[i_index])
    dipole_0 = float(parse_db_array(row.MWFN_MBIS_Atom_Dipole)[i_index, 2])
    R2       = parse_db_array(row.MWFN_MBIS_r2)
    r2       = float(R2[i_index, 0] + R2[i_index, 1] + R2[i_index, 2])

    scaffold    = row.key_value_pairs.get("Aromtic_Scaffold", "unknown")
    ring_charge = SCAFFOLD_FACTORS.get(scaffold, 0.0)

    feat_lookup[name] = dict(
        vmax=vmax, monopole=monopole, c6=c6,
        dipole_0=dipole_0, r2=r2,
        scaffold=scaffold, ring_charge=ring_charge,
    )

print(f"feat_db: {len(feat_lookup)} molecules read")

# ══════════════════════════════════════════════════════════════════════════════
# Assemble dataset (single pass over all target DBs)
# ══════════════════════════════════════════════════════════════════════════════

records = []   # list of dicts — one per molecule × target

for db_filename, (acceptor_name, atom_rule) in TARGET_DB_CONFIG.items():
    try:
        tdb = connect(db_filename)
    except Exception as e:
        print(f"WARNING: Could not open '{db_filename}': {e}")
        continue

    try:
        acc_polar, acc_mono, acc_dipole = extract_acceptor_features(
            acceptor_name, atom_rule
        )
    except Exception as e:
        print(f"WARNING: Acceptor features for '{acceptor_name}' failed: {e}")
        continue

    n_added = 0
    for row in tdb.select():
        mol_name = row.key_value_pairs.get("Name")
        if mol_name is None:
            continue
        if "Interaction_E" not in row.key_value_pairs:
            print(f"  SKIP [{db_filename}]: '{mol_name}' — no Interaction_E")
            continue
        if mol_name not in feat_lookup:
            print(f"  SKIP [{db_filename}]: '{mol_name}' — not in feat_db")
            continue

        f  = feat_lookup[mol_name]
        ie = row.Interaction_E * 627.5095   # Hartree → kcal/mol

        records.append(dict(
            name        = mol_name,
            target      = acceptor_name,
            scaffold    = f["scaffold"],
            interaction = ie,
            # Vmax model features
            vmax        = f["vmax"],
            #ring_charge = f["ring_charge"],
            # Full model extra features
            monopole    = f["monopole"],
            c6          = f["c6"],
            dipole_0    = f["dipole_0"],
            r2          = f["r2"],
            # Acceptor features (shared)
            acc_polar   = acc_polar,
            acc_mono    = acc_mono,
            acc_dipole  = acc_dipole,
        ))
        n_added += 1

    print(f"  {db_filename}: {n_added} molecules loaded")

print(f"\nTotal dataset size: {len(records)} molecules")

full_df = pd.DataFrame(records)
y       = full_df["interaction"].values
label   = full_df["name"].values

# ══════════════════════════════════════════════════════════════════════════════
# Shared train / test split  (same random state → identical splits)
# ══════════════════════════════════════════════════════════════════════════════

(df_train, df_test,
 y_train,  y_test,
 labels_train, labels_test) = train_test_split(
    full_df, y, label,
    test_size=0.2, random_state=42
)

print(f"Train: {len(df_train)} | Test: {len(df_test)}")

# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "Vmax": {
        "features": ["vmax", 
                     "acc_polar", "acc_mono", "acc_dipole"],
        "label"   : "Vmax model  [Vmax + Ring Charge + Acceptor]",
    },
    "Full": {
        "features": ["monopole", "c6", "dipole_0", "r2",
                     "acc_polar", "acc_mono", "acc_dipole"],
        "label"   : "Full model  [Monopole + C6 + Dipole + r² + Ring Charge + Acceptor]",
    },
}

def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}   # model_key → dict of fitted pipeline + predictions + metrics

for key, cfg in MODELS.items():
    feats = cfg["features"]

    X_tr = df_train[feats].values
    X_te = df_test[feats].values

    pipe = build_pipeline()

    # ── K-Fold CV ────────────────────────────────────────────────────────────
    rmse_cv = -cross_val_score(pipe, X_tr, y_train,
                               cv=kf, scoring='neg_root_mean_squared_error')
    r2_cv   =  cross_val_score(pipe, X_tr, y_train,
                               cv=kf, scoring='r2')

    print(f"\n── {key} model: Cross-Validation ─────────────────────────────────")
    print(f"RMSE per fold : {rmse_cv.round(4)}")
    print(f"Mean RMSE     : {rmse_cv.mean():.4f} ± {rmse_cv.std():.4f}")
    print(f"R² per fold   : {r2_cv.round(4)}")
    print(f"Mean R²       : {r2_cv.mean():.4f} ± {r2_cv.std():.4f}")

    # ── Per-fold weights ──────────────────────────────────────────────────────
    fold_weights = []
    for tr_idx, _ in kf.split(X_tr):
        fp = build_pipeline()
        fp.fit(X_tr[tr_idx], y_train[tr_idx])
        fold_weights.append(fp.named_steps['model'].coef_)
    fold_weights = np.array(fold_weights)

    print(f"\n── {key} model: Feature Weights Across Folds ─────────────────────")
    for i, fname in enumerate(feats):
        print(f"{fname:>24}: {fold_weights[:, i].mean():.5f} ± {fold_weights[:, i].std():.5f}")

    # ── Final fit ─────────────────────────────────────────────────────────────
    pipe.fit(X_tr, y_train)

    coefs = pipe.named_steps['model'].coef_
    bias  = pipe.named_steps['model'].intercept_

    print(f"\n── {key} model: Descriptor Coefficients ──────────────────────────")
    for fname, c in zip(feats, coefs):
        print(f"{fname:>24}: {c:.5f}")
    print(f"{'Bias':>24}: {bias:.5f}")

    # ── Predictions & metrics ─────────────────────────────────────────────────
    y_pred_tr = pipe.predict(X_tr)
    y_pred_te = pipe.predict(X_te)

    r2_train   = r2_score(y_train, y_pred_tr)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_tr))
    r2_test    = r2_score(y_test,  y_pred_te)
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_te))

    print(f"\n── {key} model: Held-Out Test Performance ────────────────────────")
    print(f"{'R²':>24}: {r2_test:.4f}")
    print(f"{'RMSE':>24}: {rmse_test:.4f} kcal/mol")
    print(f"\n── {key} model: Training Performance ────────────────────────────")
    print(f"{'R²':>24}: {r2_train:.4f}")
    print(f"{'RMSE':>24}: {rmse_train:.4f} kcal/mol")

    results[key] = dict(
        pipe         = pipe,
        feats        = feats,
        y_pred_train = y_pred_tr,
        y_pred_test  = y_pred_te,
        r2_train     = r2_train,
        rmse_train   = rmse_train,
        r2_test      = r2_test,
        rmse_test    = rmse_test,
        resid_train  = y_train - y_pred_tr,
        resid_test   = y_test  - y_pred_te,
    )

print("\n── Scaffold Ring Charges Used ────────────────────────────────────")
for scaffold, charge in sorted(SCAFFOLD_FACTORS.items()):
    print(f"{scaffold:>24}: {charge:+.4f} e")

# ══════════════════════════════════════════════════════════════════════════════
# Shared colour map (by acceptor target)
# ══════════════════════════════════════════════════════════════════════════════

all_targets = sorted(full_df["target"].unique())
palette = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22',
]
color_map = {t: palette[i % len(palette)] for i, t in enumerate(all_targets)}

targets_train = df_train["target"].to_numpy()
targets_test  = df_test["target"].to_numpy()

# ══════════════════════════════════════════════════════════════════════════════
# Plotting helper
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, stem):
    for ext in ("pdf", "svg", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"Saved → {path}")
    plt.close(fig)

def parity_ax(ax, y_tr, y_pred_tr, y_te, y_pred_te,
              tgt_tr, tgt_te, r2_train, rmse_train, r2_test, rmse_test, title):
    """Draw a parity scatter on a given Axes."""
    for tgt in all_targets:
        mask = tgt_tr == tgt
        if mask.any():
            ax.scatter(y_tr[mask], y_pred_tr[mask],
                       facecolors='none', edgecolors=color_map[tgt],
                       linewidths=1.2, s=45, zorder=2, label=f'{tgt} (train)')
    for tgt in all_targets:
        mask = tgt_te == tgt
        if mask.any():
            ax.scatter(y_te[mask], y_pred_te[mask],
                       facecolors=color_map[tgt], edgecolors='black',
                       linewidths=0.5, s=45, zorder=3, label=f'{tgt} (test)')

    all_vals = np.concatenate([y_tr, y_te, y_pred_tr, y_pred_te])
    pad  = 0.05 * (all_vals.max() - all_vals.min())
    lims = [all_vals.min() - pad, all_vals.max() + pad]
    ax.plot(lims, lims, 'k--', lw=1.0, zorder=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.set_xlabel('DFT Interaction Energy (kcal/mol)', fontsize=10)
    ax.set_ylabel('Predicted Interaction Energy (kcal/mol)', fontsize=10)
    ax.set_title(title, fontsize=11)

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
              fontsize=6.5, framealpha=0.9, loc='upper left')

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 & 2: Individual parity plots (one per model)
# ══════════════════════════════════════════════════════════════════════════════

for key, r in results.items():
    fig, ax = plt.subplots(figsize=(7, 7))
    parity_ax(ax,
              y_train, r["y_pred_train"], y_test, r["y_pred_test"],
              targets_train, targets_test,
              r["r2_train"], r["rmse_train"], r["r2_test"], r["rmse_test"],
              title=MODELS[key]["label"])
    fig.tight_layout()
    save_fig(fig, f"parity_{key.lower()}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Side-by-side parity comparison
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (key, r) in zip(axes, results.items()):
    parity_ax(ax,
              y_train, r["y_pred_train"], y_test, r["y_pred_test"],
              targets_train, targets_test,
              r["r2_train"], r["rmse_train"], r["r2_test"], r["rmse_test"],
              title=MODELS[key]["label"])
fig.suptitle("Parity Plot Comparison", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "parity_comparison")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Residual plots (predicted vs residual, train + test)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (key, r) in zip(axes, results.items()):
    # Training residuals — hollow markers
    for tgt in all_targets:
        mask = targets_train == tgt
        if mask.any():
            ax.scatter(r["y_pred_train"][mask], r["resid_train"][mask],
                       facecolors='grey',
                       linewidths=1.2, s=45, zorder=2, label='Training Set')
    # Test residuals — filled markers
    for tgt in all_targets:
        mask = targets_test == tgt
        if mask.any():
            ax.scatter(r["y_pred_test"][mask], r["resid_test"][mask],
                       facecolors=color_map[tgt], edgecolors='black',
                       linewidths=0.5, s=45, zorder=3, label=f'{tgt} (test)')

    ax.axhline(0, color='black', linestyle='--', linewidth=1.0, zorder=1)

    # Axis limits — include all residuals with padding
    all_pred_both = np.concatenate([r["y_pred_train"], r["y_pred_test"]])
    all_resid     = np.concatenate([r["resid_train"],  r["resid_test"]])
    xpad = 0.05 * (all_pred_both.max() - all_pred_both.min())
    ypad = 0.05 * (all_resid.max()     - all_resid.min())
    ax.set_xlim(all_pred_both.min() - xpad, all_pred_both.max() + xpad)
    ax.set_ylim(all_resid.min()     - ypad, all_resid.max()     + ypad)

    ax.set_xlabel('Predicted E_int (kcal/mol)', fontsize=13)
    ax.set_ylabel('Residual (DFT − Predicted) (kcal/mol)',   fontsize=13)

    sh, sl = ax.get_legend_handles_labels()
    ax.legend(handles=sh, labels=sl,
              fontsize=11, framealpha=0.9, loc='upper left')

fig.suptitle("Residual Plots", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "residual_comparison")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: KDE of residuals — both models overlaid on one panel
# ══════════════════════════════════════════════════════════════════════════════

MODEL_STYLES = {
    "Vmax": {"color": "#1f77b4", "ls": "-"},
    "Full": {"color": "#d62728", "ls": "--"},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
titles = ["Training Residuals", "Test Residuals"]
resid_keys = ["resid_train", "resid_test"]

for ax, title, rkey in zip(axes, titles, resid_keys):
    for key, r in results.items():
        resid = r[rkey]
        kde   = gaussian_kde(resid, bw_method="scott")
        xs    = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 400)
        style = MODEL_STYLES[key]
        ax.plot(xs, kde(xs),
                color=style["color"], ls=style["ls"], lw=2.0,
                label=MODELS[key]["label"])
        ax.axvline(resid.mean(),
                   color=style["color"], ls=":", lw=1.2,
                   label=f'{key} mean = {resid.mean():.3f}')

    ax.axvline(0, color='black', lw=1.0, ls='--', label='Zero residual')
    ax.set_xlabel('Residual (kcal/mol)', fontsize=10)
    ax.set_ylabel('Density',             fontsize=10)
    ax.set_title(title,                  fontsize=11)
    ax.legend(fontsize=7.5, framealpha=0.9)

fig.suptitle("KDE of Residuals: Vmax Model vs Full Model", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "kde_residuals")

# ══════════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict_vmax_model(vmax, scaffold_name, acc_polar, acc_mono, acc_dipole):
    """
    Predict interaction energy (kcal/mol) using the Vmax model.
      vmax          — Vmax on iodine from MWFN_MBIS_Iodine_Vmin_Vmax
      scaffold_name — aromatic scaffold (looked up in SCAFFOLD_FACTORS)
      acc_polar     — acceptor polarization
      acc_mono      — acceptor monopole charge
      acc_dipole    — acceptor dipole (z-component or magnitude)
    """
    ring_charge = SCAFFOLD_FACTORS.get(scaffold_name, 0.0)
    X = np.array([[vmax, ring_charge, acc_polar, acc_mono, acc_dipole]])
    return results["Vmax"]["pipe"].predict(X)[0]

def predict_full_model(monopole, c6, dipole_0, r2, scaffold_name,
                       acc_polar, acc_mono, acc_dipole):
    """
    Predict interaction energy (kcal/mol) using the Full model.
      monopole      — MBIS atomic charge on iodine
      c6            — C6 dispersion coefficient on iodine
      dipole_0      — z-component of atomic dipole on iodine
      r2            — sum of <r²> components on iodine
      scaffold_name — aromatic scaffold (looked up in SCAFFOLD_FACTORS)
      acc_polar     — acceptor polarization
      acc_mono      — acceptor monopole charge
      acc_dipole    — acceptor dipole (z-component or magnitude)
    """
    ring_charge = SCAFFOLD_FACTORS.get(scaffold_name, 0.0)
    X = np.array([[monopole, c6, dipole_0, r2, ring_charge,
                   acc_polar, acc_mono, acc_dipole]])
    return results["Full"]["pipe"].predict(X)[0]

