
from pathlib import Path
import pandas as pd

input_file = Path(
    "data/processed/multiclass_clean.csv"
)

print("Checking duplicates in cleaned dataset...")
print("=" * 70)

df = pd.read_csv(input_file, low_memory=False)

feature_columns = [
    column
    for column in df.columns
    if column not in {"Label", "Source_File"}
]

features = df[feature_columns]

# Hash feature vectors for an initial duplicate analysis.
hashes = pd.util.hash_pandas_object(
    features,
    index=False
)

df["Feature_Hash"] = hashes.to_numpy()

grouped = df.groupby("Feature_Hash").agg(
    rows=("Label", "size"),
    label_count=("Label", "nunique"),
    source_count=("Source_File", "nunique"),
)

repeated = grouped[grouped["rows"] > 1]
conflicting = repeated[repeated["label_count"] > 1]
cross_file = repeated[repeated["source_count"] > 1]

print(f"Total rows: {len(df):,}")
print(f"Unique feature hashes: {len(grouped):,}")
print(f"Repeated feature groups: {len(repeated):,}")
print(
    "Extra repeated rows:",
    f"{int((repeated['rows'] - 1).sum()):,}"
)
print(f"Groups with conflicting labels: {len(conflicting):,}")
print(f"Repeated groups across source files: {len(cross_file):,}")

print("\nDuplicate check completed.")
