
import pandas as pd
from pathlib import Path

data_path = Path("data/processed/multiclass_clean.csv")

print("Loading labels and source filenames...")

df = pd.read_csv(
    data_path,
    usecols=["Label", "Source_File"]
)

print(f"Total rows: {len(df):,}")
print(f"Number of source files: {df['Source_File'].nunique()}")
print(f"Number of classes: {df['Label'].nunique()}")

print("\nClass counts by source file:")
table = pd.crosstab(df["Label"], df["Source_File"])
print(table.to_string())

print("\nNumber of source files containing each class:")
class_coverage = (table > 0).sum(axis=1).sort_values()

print(class_coverage.to_string())

output_path = Path("data/processed/class_source_distribution.csv")
table.to_csv(output_path)

print(f"\nDistribution saved to: {output_path}")
print("\nAnalysis completed.")
