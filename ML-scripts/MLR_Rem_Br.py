from ase.db import connect
import numpy as np
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
#  CONFIGURATION
#  Change TEST_ACCEPTOR to hold out a different database for blind prediction.
#  Valid options: 'I', 'F', 'Cl', 'Br'
# =============================================================================

TEST_ACCEPTOR = 'nan-Br'

# Acceptor label -> (interaction-energy database, acceptor atomic number)
HALOGEN_DBS = {
    'nan-I':  ('I-molecules.db',  53),
    'nan-F':  ('F-molecules.db',   9),
    'nan-Cl': ('Cl-molecules.db', 17),
    'nan-Br': ('Br-molecules.db', 35),
}

FEAT_DB_PATH = 'mol.db'
ACC_DB_PATH  = 'anion-molecules.db'

# -- Scaffold ring partial charges (units: e) ---------------------------------

SCAFFOLD_FACTORS = {
    "Benzene"    : -0.666233,
    "Pyrimidine" : -0.326117,
    "BX-5"       : -0.648058,
    "BX-6"       : -0.648042,
    "BZ-5"       : -0.642792,
    "BZ-6"       : -0.642793,
}

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

# =============================================================================
#  Step 1: Load acceptor polarizabilities from acc_molecules.db
#  Each row should have key_value_pair "Name" matching one of: I, F, Cl, Br.
#  The halogen atom polarizability is extracted from that row.
# =============================================================================

print("=" * 60)
print("Step 1 - Acceptor polarizabilities")
print("=" * 60)

acc_db = connect(ACC_DB_PATH)
acc_polar_lookup = {}   # {acceptor_label: float}

for row in acc_db.select():
    name = row.key_value_pairs.get("Name", "").strip()
    if name not in HALOGEN_DBS:
        continue
    _, acc_z = HALOGEN_DBS[name]
    try:
        idx     = get_atom_index(row, acc_z)
        pol_arr = ensure_2d(parse_db_array(row.MWFN_MBIS_Atomic_Polarizability), 1)
        acc_polar_lookup[name] = pol_arr[idx, 0]
        print(f"  {name:<4}: acc_polar = {acc_polar_lookup[name]:.6f}")
    except Exception as e:
        print(f"  WARNING - {name}: {e}")

missing = [k for k in HALOGEN_DBS if k not in acc_polar_lookup]
if missing:
    raise RuntimeError(f"Could not load acceptor polarizability for: {missing}")

# =============================================================================
#  Step 2: Load interaction energies from all 4 halogen databases
# =============================================================================

print("\n" + "=" * 60)
print("Step 2 - Interaction energies")
print("=" * 60)

interaction_lookup = {}   # {acceptor_label: {mol_name: energy_kcal}}

for acceptor, (db_path, _) in HALOGEN_DBS.items():
    db   = connect(db_path)
    lkup = {}
    for row in db.select():
        mol_name = row.key_value_pairs.get("Name")
        if mol_name and "Interaction_E" in row.key_value_pairs:
            lkup[mol_name] = row.Interaction_E * 627.5095   # Eh -> kcal/mol
    interaction_lookup[acceptor] = lkup
    print(f"  {acceptor:<4} ({db_path}): {len(lkup)} molecules")

# =============================================================================
#  Step 3: Extract iodine donor descriptors from mol.db
# =============================================================================

print("\n" + "=" * 60)
print(f"Step 3 - Iodine descriptors from {FEAT_DB_PATH}")
print("=" * 60)

feat_db = connect(FEAT_DB_PATH)
donor_features = {}   # {mol_name: dict}

for row in feat_db.select():
    name = row.key_value_pairs.get("Name")
    if not name:
        continue
    try:
        i = get_iodine_index(row)

        charges  = parse_db_array(row.MWFN_MBIS_Atomic_Charges)
        c6_arr   = parse_db_array(row.MWFN_MBIS_c6)
        dip_arr  = ensure_2d(parse_db_array(row.MWFN_MBIS_Atom_Dipole), 3)
        pol_arr  = ensure_2d(parse_db_array(row.MWFN_MBIS_Atomic_Polarizability), 1)
        quad_arr = ensure_2d(parse_db_array(row.MWFN_MBIS_Atom_Quadrupole), 5)
        r2_arr   = ensure_2d(parse_db_array(row.MWFN_MBIS_r2), 3)

        scaffold    = row.key_value_pairs.get("Aromtic_Scaffold", "unknown")
        ring_charge = SCAFFOLD_FACTORS.get(scaffold, 0.0)

        donor_features[name] = {
            'monopole'   : charges[i],
            'c6'         : c6_arr[i],
            'dipole_0'   : dip_arr[i, 2],
            'r2'         : r2_arr[i, 0] + r2_arr[i, 1] + r2_arr[i, 2],
            'ring_charge': ring_charge,
            'scaffold'   : scaffold,
        }
    except Exception as e:
        print(f"  SKIP '{name}' (id={row.id}): {e}")

print(f"  {len(donor_features)} donor molecules loaded")

# =============================================================================
#  Step 4: Build combined dataset
#  Each donor molecule x acceptor pair = one row.
#  acc_polar is the only feature that varies between acceptor groups.
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
            'acceptor'   : acceptor,
            'name'       : mol_name,
            'scaffold'   : feats['scaffold'],
            'monopole'   : feats['monopole'],
            'c6'         : feats['c6'],
            'dipole_0'   : feats['dipole_0'],
            'r2'         : feats['r2'],
            'ring_charge': feats['ring_charge'],
            'acc_polar'  : acc_polar,
            'target'     : lkup[mol_name],
        })
        n += 1
    print(f"  {acceptor:<4}: {n} molecule-acceptor pairs")

df_all = pd.DataFrame(records)
print(f"\n  Total rows: {len(df_all)}")

# ── Diagnostic: show what actually landed in df_all ──────────────────────────
print("\n  Acceptor counts in combined dataset:")
print(df_all.groupby('acceptor').size().rename('n_rows').to_string())

print(f"\n  TEST_ACCEPTOR = '{TEST_ACCEPTOR}'")
print(f"  Rows matching test acceptor: {(df_all['acceptor'] == TEST_ACCEPTOR).sum()}")

# Cross-check: which Br molecule names are missing from donor_features?
br_names   = set(interaction_lookup.get(TEST_ACCEPTOR, {}).keys())
feat_names = set(donor_features.keys())
missing_in_feat = br_names - feat_names
missing_in_br   = feat_names - br_names

if missing_in_feat:
    print(f"\n  WARNING: {len(missing_in_feat)} '{TEST_ACCEPTOR}' molecule(s) "
          f"not found in mol.db:")
    for n in sorted(missing_in_feat)[:10]:
        print(f"    '{n}'")

if missing_in_br:
    print(f"\n  NOTE: {len(missing_in_br)} mol.db molecule(s) "
          f"not found in {TEST_ACCEPTOR}-molecules.db (will be skipped).")


# =============================================================================
#  Step 5: Train / test split by acceptor database (leave-one-acceptor-out)
# =============================================================================

df_train = df_all[df_all['acceptor'] != TEST_ACCEPTOR].reset_index(drop=True)
df_test  = df_all[df_all['acceptor'] == TEST_ACCEPTOR].reset_index(drop=True)

if len(df_test) == 0:
    raise RuntimeError(
        f"df_test is empty for TEST_ACCEPTOR='{TEST_ACCEPTOR}'.\n"
        f"Acceptors present in df_all: {sorted(df_all['acceptor'].unique())}\n"
        f"Check that molecule names in {TEST_ACCEPTOR}-molecules.db "
        f"match those in mol.db exactly (case, spaces, hyphens)."
     )

X_train = df_train[FEATURE_COLS].values
y_train = df_train['target'].values
X_test  = df_test[FEATURE_COLS].values
y_test  = df_test['target'].values

train_acceptors = sorted(df_train['acceptor'].unique())
print(f"\n  Train : {train_acceptors}  ({len(df_train)} rows)")
print(f"  Test  : {TEST_ACCEPTOR}  ({len(df_test)} rows - UNSEEN)")

# =============================================================================
#  Step 6: Pipeline (StandardScaler + LinearRegression)
# =============================================================================

def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ])

pipeline = build_pipeline()

# =============================================================================
#  Step 7: 5-Fold CV on training data only
# =============================================================================

print("\n" + "=" * 60)
print("Step 7 - 5-Fold Cross-Validation (training data only)")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_cv = -cross_val_score(pipeline, X_train, y_train,
                           cv=kf, scoring='neg_root_mean_squared_error')
r2_cv   =  cross_val_score(pipeline, X_train, y_train,
                           cv=kf, scoring='r2')

print(f"  RMSE per fold : {rmse_cv.round(4)}")
print(f"  Mean RMSE     : {rmse_cv.mean():.4f} +/- {rmse_cv.std():.4f}")
print(f"  R2 per fold   : {r2_cv.round(4)}")
print(f"  Mean R2       : {r2_cv.mean():.4f} +/- {r2_cv.std():.4f}")

# -- Per-fold feature weights
fold_weights = []
for train_idx, _ in kf.split(X_train):
    fp = build_pipeline()
    fp.fit(X_train[train_idx], y_train[train_idx])
    fold_weights.append(fp.named_steps['model'].coef_)

fold_weights = np.array(fold_weights)
print("\n  Feature weights across folds:")
for i, fname in enumerate(FEATURE_COLS):
    print(f"    {fname:>24}: {fold_weights[:, i].mean():.5f} +/- {fold_weights[:, i].std():.5f}")

# =============================================================================
#  Step 8: Final model on full training set
# =============================================================================

pipeline.fit(X_train, y_train)

coefs = pipeline.named_steps['model'].coef_
bias  = pipeline.named_steps['model'].intercept_

print("\n" + "=" * 60)
print("Step 8 - Final model coefficients")
print("=" * 60)
for fname, c in zip(FEATURE_COLS, coefs):
    print(f"  {fname:>24}: {c:.5f}")
print(f"  {'Bias':>24}: {bias:.5f}")

print("\n  Scaffold ring charges:")
for scaffold, charge in sorted(SCAFFOLD_FACTORS.items()):
    print(f"  {scaffold:>24}: {charge:+.4f} e")

# =============================================================================
#  Step 9: Performance metrics
# =============================================================================

y_pred_test  = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)

r2_test    = r2_score(y_test,  y_pred_test)
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
r2_train   = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

print("\n" + "=" * 60)
print("Step 9 - Performance")
print("=" * 60)
print(f"  Training ({', '.join(train_acceptors)})")
print(f"    R2   : {r2_train:.4f}")
print(f"    RMSE : {rmse_train:.4f} kcal/mol")
print(f"  Held-out test ({TEST_ACCEPTOR})")
print(f"    R2   : {r2_test:.4f}")
print(f"    RMSE : {rmse_test:.4f} kcal/mol")

# =============================================================================
#  Step 10: Parity plot — unseen test database only, coloured by scaffold
# =============================================================================

scaffolds_test = df_test['scaffold'].to_numpy()
scaffolds_train = df_train['scaffold'].to_numpy()
all_scaffolds  = sorted(set(scaffolds_test))

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
        label='Training Set',
    )


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

all_vals = np.concatenate([y_test, y_pred_test])
pad      = 0.07 * (all_vals.max() - all_vals.min())   # 7 % breathing room
lo       = all_vals.min() - pad
hi       = all_vals.max() + pad
lims     = [lo, hi]

# Perfect fit line
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
    fontsize=11.0,
    framealpha=1,
    loc='upper left',
)

fig.tight_layout()

for ext in ('pdf', 'svg', 'png'):
    path = f'parity_plot_{TEST_ACCEPTOR}.{ext}'
    fig.savefig(path, dpi=300 if ext == 'png' else None)
    print(f'  Saved -> {path}')

plt.close(fig)

# =============================================================================
#  Prediction helper
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
        Donor aromatic scaffold name (must match a key in SCAFFOLD_FACTORS,
        defaults to 0.0 e ring charge if not found).
    acceptor_label : str
        One of: 'I', 'F', 'Cl', 'Br'
    """
    ring_charge = SCAFFOLD_FACTORS.get(scaffold_name, 0.0)
    acc_polar   = acc_polar_lookup[acceptor_label]
    X = np.array([[monopole, c6, dipole_0, r2, ring_charge, acc_polar]])
    return pipeline.predict(X)[0]
