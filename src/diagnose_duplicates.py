
import pandas as pd
from pathlib import Path

file_path = Path("data/processed/ddos_clean.csv")

print("Diagnosing duplicate feature vectors...")
print("-" * 50)

df = pd.read_csv(file_path)

feature_columns = [
    column for column in df.columns
    if column != "Label"
]

# شناسه گروهی برای هر ترکیب دقیق از ویژگی‌ها
feature_hash = pd.util.hash_pandas_object(
    df[feature_columns],
    index=False
)

df["_feature_hash"] = feature_hash

# تعداد گروه‌های دارای ویژگی‌های یکسان
group_sizes = df.groupby("_feature_hash").size()

duplicate_groups = group_sizes[group_sizes > 1]

# بررسی اینکه آیا یک گروه ویژگی، چند برچسب متفاوت دارد
labels_per_group = df.groupby("_feature_hash")["Label"].nunique()

conflicting_groups = labels_per_group[labels_per_group > 1]

conflicting_rows = int(
    df["_feature_hash"].isin(conflicting_groups.index).sum()
)

print(f"Total rows: {len(df):,}")
print(f"Unique feature vectors: {group_sizes.size:,}")
print(
    "Rows belonging to repeated feature groups:",
    f"{int(group_sizes[duplicate_groups.index].sum()):,}"
)
print(
    "Extra repeated rows beyond first occurrence:",
    f"{int((duplicate_groups - 1).sum()):,}"
)
print(f"Groups with conflicting labels: {len(conflicting_groups):,}")
print(f"Rows in conflicting-label groups: {conflicting_rows:,}")

print("\nDiagnosis completed.")
