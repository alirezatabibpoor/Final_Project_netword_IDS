
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# مسیر فایل ورودی و خروجی
input_path = Path("data/processed/ddos_deduplicated.csv")
train_path = Path("data/processed/ddos_train.csv")
test_path = Path("data/processed/ddos_test.csv")

print("Loading deduplicated dataset...")

df = pd.read_csv(input_path)

# جدا کردن ویژگی‌ها از برچسب
X = df.drop(columns=["Label"])
y = df["Label"]

print(f"Total rows: {len(df):,}")

# تقسیم داده‌ها به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# بازسازی دیتافریم‌ها
train_df = X_train.copy()
train_df["Label"] = y_train

test_df = X_test.copy()
test_df["Label"] = y_test

# ذخیره فایل‌ها
train_path.parent.mkdir(parents=True, exist_ok=True)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nDataset split completed.")
print("-" * 45)
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")

print("\nTraining label distribution:")
print(train_df["Label"].value_counts())

print("\nTesting label distribution:")
print(test_df["Label"].value_counts())

print("\nSaved training dataset:", train_path)
print("Saved testing dataset:", test_path)
