"""Part B: delivery-hours regression with a leakage-safe split, diagnostics and model choice."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import DATA_DIR, FEATURES, OUTPUT_DIR, three_way_split

TREE_DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
LEARNING_RATES = [0.01, 0.1, 1.0]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred)}


OUTPUT_DIR.mkdir(exist_ok=True)
table = pd.read_csv(DATA_DIR / "modelling_table.csv")
X, y = table[FEATURES], table["delivery_hours"]
X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)

# The scaler is fitted on the training rows only, so only the training columns are centred exactly.
scaler = StandardScaler().fit(X_train)
train_mean = np.abs(scaler.transform(X_train).mean(axis=0)).max()
test_mean = np.abs(scaler.transform(X_test).mean(axis=0)).max()
pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [len(X_train), len(X_val), len(X_test), round(float(train_mean), 4), round(float(test_mean), 4)],
}, dtype=object).to_csv(OUTPUT_DIR / "part_b_checks.csv", index=False)

metric_rows = [{"model": "mean_baseline", "split": "val",
                **regression_metrics(y_val, np.full(len(y_val), y_train.mean()))}]

depth_rows = []
for depth in TREE_DEPTHS:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
    depth_rows.append({"max_depth": "unlimited" if depth is None else depth,
                       "train_rmse": rmse(y_train, tree.predict(X_train)),
                       "val_rmse": rmse(y_val, tree.predict(X_val))})
depth_table = pd.DataFrame(depth_rows)
depth_table.round(4).to_csv(OUTPUT_DIR / "part_b_tree_depth.csv", index=False)
# argmin returns the first minimum, i.e. the smaller depth on a tie.
best_depth = TREE_DEPTHS[int(depth_table["val_rmse"].to_numpy().argmin())]

candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    row = {"model": name, "split": "val", **regression_metrics(y_val, model.predict(X_val))}
    metric_rows.append(row)
    val_rmse[name] = row["RMSE"]

lr_rows = []
for lr in LEARNING_RATES:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    pred = gb.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, pred), "val_r2": r2_score(y_val, pred)})
pd.DataFrame(lr_rows).round(4).to_csv(OUTPUT_DIR / "part_b_gb_learning_rate.csv", index=False)

ranking = sorted(val_rmse, key=val_rmse.get)
X_dev, y_dev = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # one object shared by both models
cv_rows = []
for name in ranking[:2]:
    scores = cross_val_score(candidates[name], X_dev, y_dev, cv=kfold)
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
pd.DataFrame(cv_rows).round(4).to_csv(OUTPUT_DIR / "part_b_cv.csv", index=False)

# The only time the test rows are scored: the validation winner, already fitted on the training rows.
final_name = ranking[0]
metric_rows.append({"model": final_name, "split": "test",
                    **regression_metrics(y_test, candidates[final_name].predict(X_test))})
pd.DataFrame(metric_rows).round(4).to_csv(OUTPUT_DIR / "part_b_metrics.csv", index=False)

print(f"split {len(X_train)}/{len(X_val)}/{len(X_test)}; best tree depth {best_depth}; "
      f"validation ranking {ranking}; final model {final_name}")
