
from pathlib import Path
import pandas as pd

input_file = Path(
    "data/processed/multiclass_clean.csv"
)

output_file = Path(
    "data/processed/multiclass_deduplicated.csv"
)

print("Creating deduplicated multiclass dataset...")
print("=" * 70)

df = pd.read_csv(input_file, low_memory=False)

feature_columns = [
    column
    for column in df.columns
    if column not in {"Label", "Source_File"}
]

original_rows = len(df)

# Check that identical feature vectors do not have
# conflicting labels before deduplication.
label_counts = (
    df.groupby(feature_columns, dropna=False)["Label"]
    .nunique()
)

conflicting_groups = int((label_counts > 1).sum())

print(f"Original rows: {original_rows:,}")
print(
    "Feature groups with conflicting labels:",
    f"{conflicting_groups:,}"
)

if conflicting_groups > 0:
    raise ValueError(
        "Conflicting labels remain. "
        "Do not deduplicate until they are investigated."
    )

# Keep one row for each exactly identical feature vector.
# Label is excluded from the duplicate key because
# conflicts were checked above.
df_unique = df.drop_duplicates(
    subset=feature_columns,
    keep="first"
).copy()

df_unique.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print("\nDeduplication completed.")
print(f"Rows before: {original_rows:,}")
print(f"Rows after: {len(df_unique):,}")
print(f"Rows removed: {original_rows - len(df_unique):,}")

print("\nClass distribution after deduplication:")
print(df_unique["Label"].value_counts().to_string())

print(f"\nSaved to: {output_file}")
