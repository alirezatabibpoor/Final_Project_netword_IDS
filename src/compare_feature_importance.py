import pandas as pd
from pathlib import Path

dt_path = Path("reports/decision_tree_feature_importance.csv")
rf_path = Path("reports/random_forest_feature_importance.csv")
output_path = Path("reports/feature_importance_comparison.csv")

print("Loading feature importance files...")

dt = pd.read_csv(dt_path).rename(
    columns={"Importance": "Decision_Tree_Importance"}
)
rf = pd.read_csv(rf_path).rename(
    columns={"Importance": "Random_Forest_Importance"}
)

dt["Decision_Tree_Rank"] = range(1, len(dt) + 1)
rf["Random_Forest_Rank"] = range(1, len(rf) + 1)

comparison = pd.merge(
    dt[["Feature", "Decision_Tree_Importance", "Decision_Tree_Rank"]],
    rf[["Feature", "Random_Forest_Importance", "Random_Forest_Rank"]],
    on="Feature",
    how="outer"
)

comparison["Shared_Top20"] = (
    (comparison["Decision_Tree_Rank"] <= 20) &
    (comparison["Random_Forest_Rank"] <= 20)
)

comparison = comparison.sort_values(
    "Decision_Tree_Rank",
    na_position="last"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(output_path, index=False)

print("\nTop 20 Decision Tree features and their Random Forest ranks:")
print(
    comparison.sort_values("Decision_Tree_Rank")
    .head(20)
    .to_string(index=False)
)

shared = comparison[comparison["Shared_Top20"]]

print(f"\nNumber of shared top-20 features: {len(shared)}")
print("\nShared top-20 features:")
print(shared["Feature"].to_list())

print(f"\nComparison saved to: {output_path}")
print("Feature importance comparison completed.")