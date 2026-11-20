
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

DATA_PATH = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/"
    "Monday-WorkingHours.pcap_ISCX.csv"
)
MODEL_PATH = Path("models/random_forest_multiclass.joblib")

CHUNK_SIZE = 50_000
BENIGN_LABEL = "BENIGN"

print("Loading Decision Tree model...")
model = joblib.load(MODEL_PATH)

# Use exactly the same feature names and order as during training.
feature_names = list(model.feature_names_in_)

total_rows = 0
valid_rows = 0
dropped_rows = 0
false_alarms = 0
prediction_counts = {}

print("Evaluating Monday dataset in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1,
):
    # CICIDS2017 headers often contain leading/trailing spaces.
    chunk.columns = chunk.columns.str.strip()

    missing_features = [
        name for name in feature_names
        if name not in chunk.columns
    ]
    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    total_rows += len(chunk)

    X = chunk[feature_names].copy()

    # Convert features to numeric and handle invalid values.
    for column in feature_names:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    # Only evaluate rows with all required features available.
    valid_mask = X.notna().all(axis=1)
    X_valid = X.loc[valid_mask]

    dropped_rows += len(chunk) - len(X_valid)

    if X_valid.empty:
        continue

    predictions = model.predict(X_valid)
    valid_rows += len(predictions)

    for label, count in pd.Series(predictions).value_counts().items():
        prediction_counts[str(label)] = (
            prediction_counts.get(str(label), 0) + int(count)
        )

    false_alarms += int(
        np.count_nonzero(predictions != BENIGN_LABEL)
    )

    print(
        f"Processed chunk {chunk_number}: "
        f"{len(chunk):,} rows; "
        f"{len(X_valid):,} valid rows"
    )

print("\n" + "=" * 55)
print("MONDAY INDEPENDENT EVALUATION - RANDOM FOREST")
print("=" * 55)

print(f"Total input rows:       {total_rows:,}")
print(f"Valid evaluated rows:   {valid_rows:,}")
print(f"Rows excluded:          {dropped_rows:,}")
print(f"False alarms:           {false_alarms:,}")

if valid_rows:
    false_alarm_rate = false_alarms / valid_rows
    print(f"False alarm rate:       {false_alarm_rate:.6%}")
    print(
        f"Correctly predicted BENIGN: "
        f"{valid_rows - false_alarms:,}"
    )

print("\nPrediction distribution:")
for label, count in sorted(prediction_counts.items()):
    print(f"{label}: {count:,}")

print("\nEvaluation completed.")
