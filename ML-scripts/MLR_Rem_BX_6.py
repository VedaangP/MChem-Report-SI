from ase.db import connect
import numpy as np
import json
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

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
    "Benzene"       :  -0.666233,   # reference scaffold
    "Pyrimidine"   : -0.326117,
    "BX-5"      :  -0.648058,
    "BX-6"    :  -0.648042,
    "BZ-5" : -0.642792,
    "BZ-6" : -0.642793,

}

HOLDOUT_SCAFFOLD = "BX-6"   # ← train on everything else, predict on this

# ── Read database ──────────────────────────────────────────────────────────────

feat_db   = connect('mol.db')
target_db = connect('Cl-molecules.db')

interaction_e_lookup = {
    row.key_value_pairs.get("Name"): row.Interaction_E
    for row in target_db.select()
    if "Interaction_E" in row.key_value_pairs
}

features_train, y_train, labels_train = [], [], []
features_test,  y_test,  labels_test  = [], [], []
scaffolds_train, scaffolds_test        = [], []

for row in feat_db.select():
    name = row.key_value_pairs.get("Name")

    if name not in interaction_e_lookup:
        print(f"  SKIP: '{name}' — no Interaction_E, skipping")
        continue

    i_index  = get_iodine_index(row)
    scaffold = row.key_value_pairs.get("Aromtic_Scaffold", "unknown")

    monopole = parse_db_array(row.MWFN_MBIS_Atomic_Charges)[i_index]
    c6       = parse_db_array(row.MWFN_MBIS_c6)[i_index]
    dipole_0 = parse_db_array(row.MWFN_MBIS_Atom_Dipole)[i_index, 2]
    polar    = parse_db_array(row.MWFN_MBIS_Atomic_Polarizability)[i_index, 0]
    Q_0      = parse_db_array(row.MWFN_MBIS_Atom_Quadrupole)[i_index, 2]
    Q_2      = parse_db_array(row.MWFN_MBIS_Atom_Quadrupole)[i_index, 4]
    R2       = parse_db_array(row.MWFN_MBIS_r2)
    r2       = R2[i_index, 0] + R2[i_index, 1] + R2[i_index, 2]

    ring_charge = SCAFFOLD_FACTORS.get(scaffold, 0.0)
    feat_vec    = [monopole, c6, dipole_0, r2]
    target      = interaction_e_lookup[name] * 627.5095

    if scaffold == HOLDOUT_SCAFFOLD:
        features_test.append(feat_vec)
        y_test.append(target)
        labels_test.append(name)
        scaffolds_test.append(scaffold)
    else:
        features_train.append(feat_vec)
        y_train.append(target)
        labels_train.append(name)
        scaffolds_train.append(scaffold)

feature_names_num = ['Monopole', 'C6', 'Dipole_1','r2']

X_train = np.array(features_train)
X_test  = np.array(features_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)
labels_train = np.array(labels_train)
labels_test  = np.array(labels_test)

print(f"Training molecules (all except {HOLDOUT_SCAFFOLD}): {len(X_train)}")
print(f"Holdout molecules ({HOLDOUT_SCAFFOLD})            : {len(X_test)}")

if len(X_test) == 0:
    raise ValueError(f"No molecules found with scaffold '{HOLDOUT_SCAFFOLD}'. "
                     f"Check that the scaffold name matches row.Aromtic_Scaffold exactly.")

# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ])

pipeline = build_pipeline()

# ── K-Fold CV on training data ────────────────────────────────────────────────

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = -cross_val_score(pipeline, X_train, y_train,
                               cv=kf, scoring='neg_root_mean_squared_error')
r2_scores   =  cross_val_score(pipeline, X_train, y_train,
                               cv=kf, scoring='r2')

print("\n── Cross-Validation Results (training folds, excl. BX-6) ─────────")
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

# ── Predict on BX-6 holdout ───────────────────────────────────────────────────

y_pred_test = pipeline.predict(X_test)
r2_test     = r2_score(y_test, y_pred_test)
rmse_test   = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n── Holdout Performance ({HOLDOUT_SCAFFOLD}) ───────────────────────────────")
print(f"{'R²':>24}: {r2_test:.4f}")
print(f"{'RMSE':>24}: {rmse_test:.4f} kcal/mol")
print(f"\n{'Molecule':<30} {'DFT (kcal/mol)':>16} {'Pred (kcal/mol)':>16} {'Error':>10}")
print("-" * 74)
for name, dft, pred in zip(labels_test, y_test, y_pred_test):
    print(f"{name:<30} {dft:>16.4f} {pred:>16.4f} {pred - dft:>10.4f}")

# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_interaction_energy(monopole, c6, dipole_0,
                               r2, scaffold_name):
    ring_charge = SCAFFOLD_FACTORS.get(scaffold_name, 0.0)
    X = np.array([[monopole, c6, dipole_0, r2]])
    return pipeline.predict(X)[0]

# ── Parity plot ───────────────────────────────────────────────────────────────

fig = go.Figure()

# Training scaffolds (greyed out for context)
df_train_plot = pd.DataFrame(X_train, columns=feature_names_num)
df_train_plot['Scaffold'] = scaffolds_train
y_pred_train = pipeline.predict(X_train)

unique_train_scaffolds = sorted(set(scaffolds_train))
grey_shades = ['#aaaaaa', '#bbbbbb', '#cccccc', '#dddddd', '#eeeeee']

for scaffold, shade in zip(unique_train_scaffolds, grey_shades):
    mask = np.array(scaffolds_train) == scaffold
    fig.add_trace(go.Scattergl(
        x=y_train[mask],
        y=y_pred_train[mask],
        mode='markers',
        name=f'{scaffold} (train)',
        marker=dict(size=6, color=shade, line=dict(color='#999', width=0.5)),
        customdata=np.column_stack([labels_train[mask],
                                    y_train[mask].round(4),
                                    y_pred_train[mask].round(4)]),
        hovertemplate=(
            "%{customdata[0]}<br>"
            f"Scaffold: {scaffold}<br>"
            "DFT: %{customdata[1]:.3f} kcal/mol<br>"
            "Pred: %{customdata[2]:.3f} kcal/mol"
            "<extra></extra>"
        )
    ))

# BX-6 holdout (highlighted in colour)
fig.add_trace(go.Scattergl(
    x=y_test,
    y=y_pred_test,
    mode='markers',
    name=f'{HOLDOUT_SCAFFOLD} (holdout)',
    marker=dict(size=10, color='tomato', line=dict(color='black', width=0.8)),
    customdata=np.column_stack([labels_test,
                                y_test.round(4),
                                y_pred_test.round(4)]),
    hovertemplate=(
        "%{customdata[0]}<br>"
        f"Scaffold: {HOLDOUT_SCAFFOLD}<br>"
        "DFT: %{customdata[1]:.3f} kcal/mol<br>"
        "Pred: %{customdata[2]:.3f} kcal/mol"
        "<extra></extra>"
    )
))

all_vals = np.concatenate([y_train, y_test, y_pred_train, y_pred_test])
lims = [all_vals.min() - 0.5, all_vals.max() + 0.5]
fig.add_trace(go.Scatter(
    x=lims, y=lims, mode='lines',
    line=dict(color='black', dash='dash'),
    name='Perfect fit', hoverinfo='skip'
))

fig.update_layout(
    title=(
        f"Trained on all scaffolds except {HOLDOUT_SCAFFOLD} | "
        f"{HOLDOUT_SCAFFOLD} holdout: R2 = {r2_test:.3f}, RMSE = {rmse_test:.3f} kcal/mol"
    ),
    xaxis_title="DFT Interaction Energy (kcal/mol)",
    yaxis_title="Predicted Interaction Energy (kcal/mol)",
    width=750, height=750, template='plotly_white'
)

fig.write_html("parity_plot_bx6_holdout.html")


from parity_plot_mpl import save_parity_plot

y_pred_train = pipeline.predict(X_train)

save_parity_plot(
    y_train, y_pred_train, scaffolds_train, labels_train,
    y_test,  y_pred_test,  labels_test,
    r2_test, rmse_test,
    holdout_scaffold=HOLDOUT_SCAFFOLD,
    out_stem="parity_plot_bx6_holdout",
)
