
from pathlib import Path

import numpy as np
import pandas as pd

input_file = Path(
    "data/processed/multiclass_selected_raw.csv"
)

output_file = Path(
    "data/processed/multiclass_clean.csv"
)

print("Cleaning multiclass dataset...")
print("=" * 70)

df = pd.read_csv(input_file, low_memory=False)

print(f"Original rows: {len(df):,}")

feature_columns = [
    column
    for column in df.columns
    if column not in {"Label", "Source_File"}
]

# Convert all features to numeric values.
features = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace positive and negative infinity with NaN.
features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows with any invalid feature values.
valid_rows = features.notna().all(axis=1)

invalid_count = int((~valid_rows).sum())

df = df.loc[valid_rows].copy()
features = features.loc[valid_rows].copy()

print(f"Rows removed for invalid features: {invalid_count:,}")
print(f"Rows remaining after feature cleaning: {len(df):,}")

# Detect feature vectors associated with multiple labels.
print("\nChecking conflicting labels...")

feature_hashes = pd.util.hash_pandas_object(
    features,
    index=False
)

df["__feature_hash"] = feature_hashes.to_numpy()

label_counts = (
    df.groupby("__feature_hash")["Label"]
    .nunique()
)

conflicting_hashes = label_counts[
    label_counts > 1
].index

conflicting_rows = df["__feature_hash"].isin(
    conflicting_hashes
)

conflicting_row_count = int(conflicting_rows.sum())

print(f"Conflicting feature groups: {len(conflicting_hashes):,}")
print(f"Rows excluded for conflicting labels: {conflicting_row_count:,}")

# Exclude every row belonging to a conflicting group.
df = df.loc[~conflicting_rows].copy()

# Remove the internal diagnostic column.
df = df.drop(columns=["__feature_hash"])

# Save the cleaned dataset.
df.to_csv(output_file, index=False, encoding="utf-8")

print("\nFinal dataset")
print("-" * 70)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Classes: {df['Label'].nunique()}")

print("\nLabel distribution:")
print(df["Label"].value_counts().to_string())

print(f"\nSaved to: {output_file}")
print("\nCleaning completed.")
