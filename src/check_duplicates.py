
import pandas as pd
from pathlib import Path

train_path = Path("data/processed/ddos_train.csv")
test_path = Path("data/processed/ddos_test.csv")

print("Checking overlap between training and testing data...")
print("-" * 55)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

feature_columns = [
    column for column in train_df.columns
    if column != "Label"
]

# محاسبه شناسه برای ترکیب ویژگی‌های هر ردیف
train_hashes = pd.util.hash_pandas_object(
    train_df[feature_columns],
    index=False
)

test_hashes = pd.util.hash_pandas_object(
    test_df[feature_columns],
    index=False
)

# شناسه‌های موجود در آموزش
train_hash_set = set(train_hashes)

# شمارش نمونه‌های آزمون که ویژگی‌هایشان در آموزش هم وجود دارد
overlap_count = int(test_hashes.isin(train_hash_set).sum())

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")
print(f"Test rows with matching feature vectors in training: {overlap_count:,}")

print("\nCheck completed.")
