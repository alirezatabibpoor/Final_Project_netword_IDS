
import pandas as pd
from pathlib import Path

input_path = Path("data/processed/ddos_clean.csv")
output_path = Path("data/processed/ddos_deduplicated.csv")

print("Loading cleaned dataset...")

df = pd.read_csv(input_path)

original_rows = len(df)

# چون هیچ گروهی برچسب متناقض ندارد،
# نگه‌داشتن یک نمونه از هر بردار ویژگی کافی است.
df = df.drop_duplicates(
    subset=[column for column in df.columns if column != "Label"],
    keep="first"
)

deduplicated_rows = len(df)
removed_rows = original_rows - deduplicated_rows

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nDeduplication completed.")
print("-" * 45)
print(f"Original rows: {original_rows:,}")
print(f"Rows after deduplication: {deduplicated_rows:,}")
print(f"Removed duplicate rows: {removed_rows:,}")
print(f"Saved to: {output_path}")
print("\nLabel distribution:")
print(df["Label"].value_counts())
