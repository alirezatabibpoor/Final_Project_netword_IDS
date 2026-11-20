
import pandas as pd
from pathlib import Path

data_path = Path("data/processed/multiclass_clean.csv")

print("Loading dataset...")
df = pd.read_csv(data_path)

feature_columns = [
    column for column in df.columns
    if column not in ["Label", "Source_File"]
]

print(f"Total rows: {len(df):,}")
print(f"Feature columns: {len(feature_columns)}")

print("\nCounting unique source files per feature vector...")

source_counts = (
    df.groupby(feature_columns, dropna=False)["Source_File"]
    .nunique()
)

cross_file_groups = source_counts[source_counts > 1]

print(f"Unique feature vectors: {len(source_counts):,}")
print(f"Feature vectors appearing in multiple source files: "
      f"{len(cross_file_groups):,}")

print("\nCounting rows belonging to cross-file feature groups...")

cross_file_mask = df.set_index(feature_columns).index.isin(
    cross_file_groups.index
)

print(f"Rows belonging to cross-file groups: "
      f"{int(cross_file_mask.sum()):,}")

print("\nCross-file duplicate analysis completed.")
