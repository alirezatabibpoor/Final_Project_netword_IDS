
from pathlib import Path

import pandas as pd

input_file = Path(
    "data/processed/multiclass_selected_raw.csv"
)

print("Verifying the largest conflicting group...")
print("=" * 70)

df = pd.read_csv(input_file, low_memory=False)

feature_columns = [
    column
    for column in df.columns
    if column not in {"Label", "Source_File"}
]

# Normalize numeric values for locating the suspicious group.
features = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

hashes = pd.util.hash_pandas_object(
    features,
    index=False
)

df["feature_hash"] = hashes.to_numpy()

label_counts = df.groupby("feature_hash")["Label"].nunique()
conflicting_hashes = label_counts[label_counts > 1].index

conflicts = df[
    df["feature_hash"].isin(conflicting_hashes)
]

group_sizes = conflicts["feature_hash"].value_counts()
largest_hash = group_sizes.index[0]

group = conflicts[
    conflicts["feature_hash"] == largest_hash
].copy()

print(f"\nLargest group size: {len(group):,}")

print("\nLabel counts:")
print(group["Label"].value_counts().to_string())

print("\nSource file counts:")
print(group["Source_File"].value_counts().to_string())

# Count features with more than one distinct normalized value.
group_features = features.loc[group.index]

different_columns = []

for column in feature_columns:
    if group_features[column].nunique(dropna=False) > 1:
        different_columns.append(column)

print(
    "\nFeature columns with differing normalized values:",
    len(different_columns)
)

if different_columns:
    print(different_columns[:30])
else:
    print("None")

print("\nMissing values in this group's features:")
missing_counts = group_features.isna().sum()
missing_counts = missing_counts[missing_counts > 0]

if missing_counts.empty:
    print("None")
else:
    print(missing_counts.to_string())

# Show a compact sample of the suspicious rows.
preview_columns = [
    "Label",
    "Source_File",
    *feature_columns[:8],
]

print("\nFirst 10 rows (selected columns):")
print(group[preview_columns].head(10).to_string(index=False))

print("\nVerification completed.")
