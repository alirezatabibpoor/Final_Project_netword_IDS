
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

DATA_PATH = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)

MODELS = {
    "Decision Tree": Path("models/decision_tree_baseline.joblib"),
    "Random Forest": Path("models/random_forest_multiclass.joblib"),
}

CHUNK_SIZE = 25_000

print("Loading models...")
models = {
    name: joblib.load(path)
    for name, path in MODELS.items()
}

# Both models must use the same feature set and order.
dt_features = list(models["Decision Tree"].feature_names_in_)
rf_features = list(models["Random Forest"].feature_names_in_)

if dt_features != rf_features:
    raise ValueError("The models do not share the same feature schema.")

feature_names = dt_features

# Each model has a binary confusion matrix:
# TP = attack correctly predicted as attack
# FN = attack predicted as BENIGN
# FP = BENIGN predicted as attack
# TN = BENIGN correctly predicted as BENIGN
results = {
    name: {
        "TP": 0,
        "FN": 0,
        "FP": 0,
        "TN": 0,
    }
    for name in models
}

total_rows = 0
valid_rows = 0
dropped_rows = 0
true_benign = 0
true_attacks = 0
attack_label_counts = {}

print("Evaluating WebAttacks dataset in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1,
):
    chunk.columns = chunk.columns.str.strip()

    if "Label" not in chunk.columns:
        raise ValueError("Label column not found.")

    missing_features = [
        name for name in feature_names
        if name not in chunk.columns
    ]
    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    total_rows += len(chunk)

    labels = chunk["Label"].fillna("").astype(str).str.strip()
    y_true_attack = labels.ne("BENIGN").to_numpy()

    X = chunk[feature_names].copy()

    for column in feature_names:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    valid_mask = X.notna().all(axis=1)

    X_valid = X.loc[valid_mask]
    y_valid = y_true_attack[valid_mask.to_numpy()]
    valid_labels = labels.loc[valid_mask]

    dropped_rows += len(chunk) - len(X_valid)

    if X_valid.empty:
        continue

    valid_rows += len(X_valid)
    true_attacks += int(y_valid.sum())
    true_benign += int((~y_valid).sum())

    attack_counts = valid_labels[valid_labels != "BENIGN"].value_counts()
    for label, count in attack_counts.items():
        attack_label_counts[label] = (
            attack_label_counts.get(label, 0) + int(count)
        )

    for name, model in models.items():
        predictions = model.predict(X_valid)

        # Any model class other than BENIGN is treated as an alarm.
        y_pred_attack = predictions != "BENIGN"

        results[name]["TP"] += int(
            np.sum(y_valid & y_pred_attack)
        )
        results[name]["FN"] += int(
            np.sum(y_valid & ~y_pred_attack)
        )
        results[name]["FP"] += int(
            np.sum(~y_valid & y_pred_attack)
        )
        results[name]["TN"] += int(
            np.sum(~y_valid & ~y_pred_attack)
        )

    print(
        f"Processed chunk {chunk_number}: "
        f"{len(chunk):,} rows; "
        f"{len(X_valid):,} valid rows"
    )

print("\n" + "=" * 60)
print("BINARY EVALUATION: UNSEEN WEB ATTACKS")
print("=" * 60)

print(f"Total input rows:       {total_rows:,}")
print(f"Valid evaluated rows:   {valid_rows:,}")
print(f"Rows excluded:          {dropped_rows:,}")
print(f"Valid BENIGN rows:      {true_benign:,}")
print(f"Valid attack rows:      {true_attacks:,}")

print("\nAttack labels in evaluated rows:")
for label, count in sorted(attack_label_counts.items()):
    print(f"{label}: {count:,}")

for name, cm in results.items():
    tp, fn = cm["TP"], cm["FN"]
    fp, tn = cm["FP"], cm["TN"]

    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
    )

    print(f"\n--- {name} ---")
    print(f"TP (attacks detected): {tp:,}")
    print(f"FN (attacks missed):   {fn:,}")
    print(f"FP (false alarms):     {fp:,}")
    print(f"TN (BENIGN correct):   {tn:,}")
    print(f"Attack recall:         {recall:.6%}")
    print(f"Attack precision:      {precision:.6%}")
    print(f"False alarm rate:      {fpr:.6%}")
    print(f"F1 score:              {f1:.6%}")
    print(f"Accuracy:              {accuracy:.6%}")

print("\nEvaluation completed.")
