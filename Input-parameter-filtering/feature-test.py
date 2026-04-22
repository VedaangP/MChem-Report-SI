import os
from ase.db import connect
from sklearn.model_selection import cross_validate
from itertools import combinations
import numpy as np
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score


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


feat_db   = connect('mol.db')
target_db = connect('Cl-molecules.db')
acc_db = connect('acc_molecules.db')

# ── Extract acceptor symbol from target_db filename ───────────────────────────
# 'Cl-molecules.db' → 'Cl',  'Br-molecules.db' → 'Br', etc.
target_db_name  = os.path.basename(target_db.filename)        # e.g. 'Cl-molecules.db'
acc_symbol      = target_db_name.split('-')[0]                 # e.g. 'Cl'

# ── Look up acceptor properties once from acc_db ──────────────────────────────
acc_rows = list(acc_db.select(Name=acc_symbol))
if len(acc_rows) == 0:
    raise ValueError(f"No row found in acc_db for acceptor '{acc_symbol}'")
if len(acc_rows) > 1:
    print(f"  WARNING: multiple rows for '{acc_symbol}' in acc_db — using first")

acc_row   = acc_rows[0]
acc_index = 0                                                  # adjust if needed

R2_acc    = parse_db_array(acc_row.MWFN_MBIS_r2)
r2_acc    = R2_acc[acc_index, 0] + R2_acc[acc_index, 1] + R2_acc[acc_index, 2]
polar_acc = parse_db_array(acc_row.MWFN_MBIS_Atomic_Polarizability)[acc_index, 0]

print(f"Acceptor: {acc_symbol}  |  r2_acc={r2_acc:.4f}  polar_acc={polar_acc:.4f}")

interaction_e_lookup = {row.Name: row.Interaction_E for row in target_db.select()}

ALL_FEATURE_NAMES = ['Monopole', 'Dipole_1', 'Q_0', 'Q_2', 'C6', 'r2', 'r2_acc','Polarizability','Polar_acc']

features, targets, labels, scaffolds = [], [], [], []

for row in feat_db.select():
    name    = row.Name
    i_index = get_iodine_index(row)

    monopole = parse_db_array(row.MWFN_MBIS_Atomic_Charges)[i_index]
    c6       = parse_db_array(row.MWFN_MBIS_c6)[i_index]
    dipole_0 = parse_db_array(row.MWFN_MBIS_Atom_Dipole)[i_index, 2]
    polar    = parse_db_array(row.MWFN_MBIS_Atomic_Polarizability)[i_index, 0]
    Q_0      = parse_db_array(row.MWFN_MBIS_Atom_Quadrupole)[i_index, 2]
    Q_2      = parse_db_array(row.MWFN_MBIS_Atom_Quadrupole)[i_index, 4]
    R2       = parse_db_array(row.MWFN_MBIS_r2)
    r2       = R2[i_index, 0] + R2[i_index, 1] + R2[i_index, 2]

    # Order must match ALL_FEATURE_NAMES
    features.append([monopole, dipole_0, Q_0, Q_2, c6, r2,r2_acc, polar,polar_acc])
    targets.append(interaction_e_lookup[name] * 627.5095)
    labels.append(name)
    scaffolds.append(row.Aromtic_Scaffold)

print(f"Database read: {len(features)} molecules")

y     = np.array(targets)
label = np.array(labels)

df_all = pd.DataFrame(features, columns=ALL_FEATURE_NAMES)
df_all['Scaffold'] = scaffolds


# ── Train/test split (done once) ───────────────────────────────────────────────

(df_train_all, df_test_all,
 y_train, y_test,
 labels_train, labels_test) = train_test_split(
    df_all, y, label,
    test_size=0.2, random_state=42
)

print(f"Train: {len(df_train_all)}  |  Test: {len(df_test_all)}")


# ── Pipeline builder ───────────────────────────────────────────────────────────

def build_pipeline(active_features):
    preprocessor = ColumnTransformer(transformers=[
        ('scale',  StandardScaler(),                                   active_features),
        ('encode', OneHotEncoder(drop='first', sparse_output=False,
                                 handle_unknown='ignore'), ['Scaffold']),
    ])
    return Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])


# ── Diagnostic helpers ─────────────────────────────────────────────────────────

def compute_aic_bic(y_true, y_pred, n_params):
    n      = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    log_ll = -n / 2 * np.log(ss_res / n)
    return 2 * n_params - 2 * log_ll, n_params * np.log(n) - 2 * log_ll


def compute_adj_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def run_model(active_features, df_train, df_test, y_train, y_test, kf):
    cols     = active_features + ['Scaffold']
    df_tr    = df_train[cols]
    df_te    = df_test[cols]
    pipeline = build_pipeline(active_features)

    # CV scores (train + val)
    cv_res = cross_validate(pipeline, df_tr, y_train, cv=kf,
                             scoring='r2', return_train_score=True)
    train_r2_cv = cv_res['train_score'].mean()
    val_r2_cv   = cv_res['test_score'].mean()
    cv_rmse     = (-cross_val_score(pipeline, df_tr, y_train,
                                     cv=kf,
                                     scoring='neg_root_mean_squared_error')).mean()

    # Per-fold weights for stability
    fold_weights = []
    for train_idx, _ in kf.split(df_tr):
        fp = build_pipeline(active_features)
        fp.fit(df_tr.iloc[train_idx], y_train[train_idx])
        fold_weights.append(fp.named_steps['model'].coef_)
    fold_weights = np.array(fold_weights)

    # Final model on all training data
    pipeline.fit(df_tr, y_train)
    ohe_final           = pipeline.named_steps['preprocessor'].named_transformers_['encode']
    scaffold_coef_names = ohe_final.get_feature_names_out(['Scaffold']).tolist()
    feature_names_all   = active_features + scaffold_coef_names

    # In-sample metrics
    y_pred_train = pipeline.predict(df_tr)
    r2_train     = r2_score(y_train, y_pred_train)
    n_params     = len(feature_names_all) + 1
    adj_r2       = compute_adj_r2(r2_train, len(df_tr), n_params)
    aic, bic     = compute_aic_bic(y_train, y_pred_train, n_params)

    # Test metrics
    y_pred_test = pipeline.predict(df_te)
    r2_test     = r2_score(y_test, y_pred_test)
    rmse_test   = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Coefficient stability
    unstable = []
    for i, name in enumerate(feature_names_all):
        mean = fold_weights[:, i].mean()
        std  = fold_weights[:, i].std()
        if abs(mean) > 1e-10 and std / abs(mean) > 0.5:
            unstable.append(name)

    return {
        'Train_R2_CV': round(train_r2_cv, 4),
        'Val_R2_CV':   round(val_r2_cv,   4),
        'Gap':         round(train_r2_cv - val_r2_cv, 4),
        'CV_RMSE':     round(cv_rmse,      4),
        'Adj_R2':      round(adj_r2,       4),
        'AIC':         round(aic,          2),
        'BIC':         round(bic,          2),
        'Test_R2':     round(r2_test,      4),
        'Test_RMSE':   round(rmse_test,    4),
        'Unstable':    ', '.join(unstable) if unstable else '—',
    }


# ── Iterate over ALL combinations ─────────────────────────────────────────────

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Total: sum(C(7,k) for k=1..7) = 127 combinations
all_combos = []
for size in range(1, len(ALL_FEATURE_NAMES) + 1):
    for combo in combinations(ALL_FEATURE_NAMES, size):
        all_combos.append(list(combo))

total = len(all_combos)
print(f"\nRunning {total} feature combinations...\n")

summary_rows = []
for model_id, active_features in enumerate(all_combos, start=1):
    metrics = run_model(active_features, df_train_all, df_test_all,
                        y_train, y_test, kf)
    row = {
        'Model_ID':   model_id,
        'N_features': len(active_features),
        'Features':   ', '.join(active_features),
        **metrics,
    }
    summary_rows.append(row)

    # Progress indicator every 10 models
    if model_id % 10 == 0 or model_id == total:
        print(f"  [{model_id}/{total}] {', '.join(active_features)}"
              f"  →  Val R²={metrics['Val_R2_CV']:.4f}  Test R²={metrics['Test_R2']:.4f}")

# ── Save and summarise ─────────────────────────────────────────────────────────

summary_df = pd.DataFrame(summary_rows)

# Sort by Val_R2_CV descending so best models appear first
summary_df = summary_df.sort_values('Val_R2_CV', ascending=False).reset_index(drop=True)
summary_df['Rank'] = summary_df.index + 1

summary_df.to_csv("feature_selection_all_combinations.csv", index=False)
print(f"\nSaved → feature_selection_all_combinations.csv  ({total} models)")

# Print top 10 by Val R²
print("\n── Top 10 models by Val R² ───────────────────────────────────────")
top10 = summary_df.head(10)[['Rank', 'N_features', 'Features',
                               'Val_R2_CV', 'Test_R2', 'Gap', 'AIC', 'BIC',
                               'Test_RMSE', 'Unstable']]
print(top10.to_string(index=False))
