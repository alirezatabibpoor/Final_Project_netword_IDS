
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


def normalize_label(label):
    """Normalize label text for grouping attack types."""
    label = str(label).strip().lower()

    if label == "benign":
        return "BENIGN"
    if "brute force" in label:
        return "Web Attack - Brute Force"
    if "xss" in label:
        return "Web Attack - XSS"
    if "sql injection" in label:
        return "Web Attack - Sql Injection"

    return label


print("Loading models...")
models = {
    name: joblib.load(path)
    for name, path in MODELS.items()
}

feature_names = list(
    models["Decision Tree"].feature_names_in_
)

if feature_names != list(
    models["Random Forest"].feature_names_in_
):
    raise ValueError("Model feature schemas do not match.")

attack_types = [
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
]

results = {
    model_name: {
        attack: {"total": 0, "detected": 0}
        for attack in attack_types
    }
    for model_name in models
}

dropped_rows = 0
total_rows = 0

print("Analyzing attack types in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1,
):
    chunk.columns = chunk.columns.str.strip()

    if "Label" not in chunk.columns:
        raise ValueError("Label column not found.")

    missing = [
        feature for feature in feature_names
        if feature not in chunk.columns
    ]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    total_rows += len(chunk)

    labels = chunk["Label"].fillna("").map(normalize_label)

    X = chunk[feature_names].copy()

    for column in feature_names:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    valid_mask = X.notna().all(axis=1)

    dropped_rows += int((~valid_mask).sum())

    X_valid = X.loc[valid_mask]
    labels_valid = labels.loc[valid_mask]

    if X_valid.empty:
        continue

    # Count true examples for each attack type.
    for attack in attack_types:
        results_total = int((labels_valid == attack).sum())

        for model_name in models:
            results[model_name][attack]["total"] += results_total

    for model_name, model in models.items():
        predictions = pd.Series(
            model.predict(X_valid),
            index=X_valid.index,
        ).astype(str).str.strip().str.lower()

        # A prediction is an alarm if it is not BENIGN.
        predicted_attack = predictions != "benign"

        for attack in attack_types:
            actual_attack = labels_valid == attack
            detected = int((actual_attack & predicted_attack).sum())

            results[model_name][attack]["detected"] += detected

    print(
        f"Processed chunk {chunk_number}: "
        f"{len(chunk):,} rows"
    )

print("\n" + "=" * 65)
print("PER-ATTACK DETECTION RESULTS")
print("=" * 65)
print(f"Total rows:   {total_rows:,}")
print(f"Rows excluded: {dropped_rows:,}")

for model_name in models:
    print(f"\n--- {model_name} ---")

    for attack in attack_types:
        total = results[model_name][attack]["total"]
        detected = results[model_name][attack]["detected"]
        missed = total - detected
        recall = detected / total if total else 0.0

        print(f"\n{attack}")
        print(f"Actual samples: {total:,}")
        print(f"Detected:       {detected:,}")
        print(f"Missed:         {missed:,}")
        print(f"Detection rate: {recall:.4%}")

print("\nAnalysis completed.")
