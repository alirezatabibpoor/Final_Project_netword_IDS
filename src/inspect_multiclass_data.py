
from pathlib import Path

import numpy as np
import pandas as pd

input_file = Path(
    "data/processed/multiclass_selected_raw.csv"
)

print("Inspecting multiclass dataset...")
print("=" * 70)

# Load the dataset
df = pd.read_csv(input_file, low_memory=False)

print(f"\nDataset shape: {df.shape}")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Check required columns
required_columns = {"Label", "Source_File"}
missing_required = required_columns - set(df.columns)

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

# Identify feature columns
feature_columns = [
    column
    for column in df.columns
    if column not in {"Label", "Source_File"}
]

# Convert feature columns to numeric for inspection only
numeric_features = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# Count missing and non-numeric values
missing_counts = numeric_features.isna().sum()

print("\nColumns with missing or non-numeric values:")
invalid_missing = missing_counts[missing_counts > 0]

if invalid_missing.empty:
    print("None")
else:
    print(invalid_missing.sort_values(ascending=False))

# Count infinite values
infinite_counts = pd.Series(
    np.isinf(numeric_features.to_numpy()).sum(axis=0),
    index=feature_columns
)

print("\nColumns with infinite values:")
infinite_columns = infinite_counts[infinite_counts > 0]

if infinite_columns.empty:
    print("None")
else:
    print(infinite_columns.sort_values(ascending=False))

# Check duplicate rows
print("\nDuplicate analysis:")
print(
    "Exact duplicate rows (including label and source):",
    f"{df.duplicated().sum():,}"
)

print(
    "Duplicate feature vectors (excluding label and source):",
    f"{df.duplicated(subset=feature_columns).sum():,}"
)

# Class distribution
print("\nLabel distribution:")
print(df["Label"].value_counts().to_string())

# Source file distribution
print("\nSource file distribution:")
print(df["Source_File"].value_counts().to_string())

print("\nInspection completed.")
