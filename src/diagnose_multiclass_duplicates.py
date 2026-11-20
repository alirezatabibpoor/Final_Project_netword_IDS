
from pathlib import Path

import pandas as pd

input_file = Path(
    "data/processed/multiclass_selected_raw.csv"
)

print("Diagnosing multiclass duplicates...")
print("=" * 70)

df = pd.read_csv(input_file, low_memory=False)

feature_columns = [
    column
    for column in df.columns
    if column not in {"Label", "Source_File"}
]

# Convert features consistently for duplicate diagnosis.
features = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# Create a hash for each feature vector.
# This avoids grouping directly on all 78 feature columns.
print("Calculating feature-vector hashes...")

feature_hash = pd.util.hash_pandas_object(
    features,
    index=False
)

diagnosis = pd.DataFrame({
    "feature_hash": feature_hash,
    "Label": df["Label"].astype("string"),
    "Source_File": df["Source_File"].astype("string"),
})

print("Grouping repeated feature vectors...")

grouped = diagnosis.groupby("feature_hash").agg(
    row_count=("Label", "size"),
    label_count=("Label", "nunique"),
    source_count=("Source_File", "nunique"),
)

repeated = grouped[grouped["row_count"] > 1]

conflicting = repeated[repeated["label_count"] > 1]
cross_file = repeated[repeated["source_count"] > 1]

print("\nDuplicate diagnosis")
print("-" * 70)

print(f"Total rows: {len(df):,}")
print(f"Unique feature hashes: {grouped.shape[0]:,}")
print(f"Repeated feature groups: {len(repeated):,}")
print(
    "Extra repeated rows beyond first occurrence:",
    f"{int((repeated['row_count'] - 1).sum()):,}"
)
print(f"Groups with conflicting labels: {len(conflicting):,}")
print(
    "Rows in conflicting-label groups:",
    f"{int(conflicting['row_count'].sum()):,}"
)
print(f"Repeated groups across multiple source files: {len(cross_file):,}")

print("\nMost frequent repeated labels in the dataset:")
print(df["Label"].value_counts().to_string())

print("\nDiagnosis completed.")
