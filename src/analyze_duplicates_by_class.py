
from pathlib import Path
import pandas as pd

input_file = Path(
    "data/processed/multiclass_clean.csv"
)

print("Analyzing duplicates by class...")
print("=" * 70)

df = pd.read_csv(input_file, low_memory=False)

feature_columns = [
    column
    for column in df.columns
    if column not in {"Label", "Source_File"}
]

features = df[feature_columns]

print("Calculating feature hashes...")

df["Feature_Hash"] = pd.util.hash_pandas_object(
    features,
    index=False
).to_numpy()

# Count original rows per class.
row_counts = df.groupby("Label").size()

# Count unique feature vectors per class.
unique_vectors = (
    df.drop_duplicates(subset=["Feature_Hash"])
    .groupby("Label")
    .size()
)

# Summarize duplication by class.
summary = pd.DataFrame({
    "Original_Rows": row_counts,
    "Unique_Feature_Vectors": unique_vectors,
})

summary["Extra_Duplicate_Rows"] = (
    summary["Original_Rows"]
    - summary["Unique_Feature_Vectors"]
)

summary["Duplicate_Percentage"] = (
    summary["Extra_Duplicate_Rows"]
    / summary["Original_Rows"]
    * 100
).round(2)

summary = summary.sort_values(
    "Original_Rows",
    ascending=False
)

print("\nDuplicate statistics by class:")
print(summary.to_string())

print("\nTotal original rows:", f"{len(df):,}")
print(
    "Total unique feature vectors:",
    f"{df['Feature_Hash'].nunique():,}"
)

print("\nAnalysis completed.")
