
from pathlib import Path

import pandas as pd

input_file = Path(
    "data/processed/multiclass_selected_raw.csv"
)

print("Inspecting conflicting-label groups...")
print("=" * 70)

df = pd.read_csv(input_file, low_memory=False)

feature_columns = [
    column
    for column in df.columns
    if column not in {"Label", "Source_File"}
]

features = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

print("Calculating feature hashes...")

df["feature_hash"] = pd.util.hash_pandas_object(
    features,
    index=False
).to_numpy()

# Find feature hashes associated with multiple labels.
label_counts = (
    df.groupby("feature_hash")["Label"]
    .nunique()
)

conflicting_hashes = label_counts[
    label_counts > 1
].index

conflicts = df[
    df["feature_hash"].isin(conflicting_hashes)
].copy()

print(f"\nConflicting groups: {len(conflicting_hashes):,}")
print(f"Rows in those groups: {len(conflicts):,}")

# Summarize labels and source files in each group.
summary = (
    conflicts.groupby("feature_hash")
    .agg(
        rows=("Label", "size"),
        labels=("Label", lambda s: sorted(s.unique().tolist())),
        sources=(
            "Source_File",
            lambda s: sorted(s.unique().tolist())
        ),
    )
    .reset_index(drop=True)
)

print("\nFirst 15 conflicting groups:")
print(summary.head(15).to_string(index=False))

print("\nConflicting label-pair counts:")
pairs = (
    conflicts.groupby("feature_hash")["Label"]
    .apply(lambda s: " | ".join(sorted(s.unique())))
    .value_counts()
)

print(pairs.to_string())

print("\nInspection completed.")
