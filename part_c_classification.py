"""Part C: delay classifier, feature-signal check, forest tuning and the operations alert threshold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import DATA_DIR, FEATURES, NOISE_COLUMNS, OUTPUT_DIR, three_way_split

TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10]
SELECT_KS = [5, 7, 10, 12, 17]
SELECTOR_K = 7
PARAM_GRID = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
DEFAULT_PARAMS = {"max_depth": None, "min_samples_leaf": 1, "n_estimators": 100}
# Exact decimals: np.arange(0.1, 1.0, 0.1) would give 0.30000000000000004 and shift the boundary.
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_RECALL = 0.80


def scorecard(y_true, y_pred):
    """Confusion-matrix cells and the rates derived from them; precision is 0 when nothing is flagged."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


OUTPUT_DIR.mkdir(exist_ok=True)
table = pd.read_csv(DATA_DIR / "modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]
X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, stratify=True)
print(f"split {len(X_train)}/{len(X_val)}/{len(X_test)}, delayed {y_train.sum()}/{y_val.sum()}/{y_test.sum()}")

# Task 2: blind baseline and three classifiers at the default 0.5 rule.
model_rows = [{"model": "blind_all_on_time", **scorecard(y_val, np.zeros(len(y_val), dtype=int))}]
logistic = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **scorecard(y_val, logistic.predict(X_val))})

best_depth, best_tree_f1, best_tree_pred = None, -1.0, None
for depth in TREE_DEPTHS:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42).fit(X_train, y_train)
    pred = tree.predict(X_val)
    tree_f1 = f1_score(y_val, pred)
    if tree_f1 > best_tree_f1:  # strict '>' keeps the smaller depth on a tie
        best_depth, best_tree_f1, best_tree_pred = depth, tree_f1, pred
model_rows.append({"model": f"decision_tree_depth_{best_depth}", **scorecard(y_val, best_tree_pred)})

forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **scorecard(y_val, forest.predict(X_val))})
pd.DataFrame(model_rows).round(4).to_csv(OUTPUT_DIR / "part_c_models.csv", index=False)

# Task 3: which columns carry signal? `forest` is the default forest on all 17 training columns.
f_scores = SelectKBest(f_classif, k="all").fit(X_train, y_train).scores_
feature_scores = pd.DataFrame({"feature": FEATURES, "f_score": f_scores, "rf_importance": forest.feature_importances_})
feature_scores["f_rank"] = feature_scores["f_score"].rank(ascending=False, method="first").astype(int)
feature_scores["is_noise"] = feature_scores["feature"].isin(NOISE_COLUMNS)
feature_scores = feature_scores.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
feature_scores.round(4).to_csv(OUTPUT_DIR / "part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in SELECT_KS:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
pd.DataFrame(selectk_rows).round(4).to_csv(OUTPUT_DIR / "part_c_selectk.csv", index=False)

# Task 4: tune on the 7 selected columns; the selector is fitted on training rows only, test rows untouched.
selector = SelectKBest(f_classif, k=SELECTOR_K).fit(X_train, y_train)
X_train_k, X_val_k = selector.transform(X_train), selector.transform(X_val)
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), PARAM_GRID, cv=5).fit(X_train_k, y_train)
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), PARAM_GRID, n_iter=9, cv=5,
                                   random_state=42).fit(X_train_k, y_train)
default_forest = RandomForestClassifier(**DEFAULT_PARAMS, random_state=42).fit(X_train_k, y_train)

tuning_rows = []
for name, search in [("grid_search", grid_search), ("randomized_search", random_search)]:
    n_candidates = len(search.cv_results_["params"])
    tuning_rows.append({"search": name, "candidates": n_candidates, "cv_folds": search.n_splits_,
                        "total_fits": n_candidates * search.n_splits_, "best_params": str(search.best_params_),
                        "best_cv_accuracy": search.best_score_,
                        "val_f1": f1_score(y_val, search.best_estimator_.predict(X_val_k))})
tuning_rows.append({"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
                    "best_params": str(DEFAULT_PARAMS), "best_cv_accuracy": np.nan,
                    "val_f1": f1_score(y_val, default_forest.predict(X_val_k))})
pd.DataFrame(tuning_rows).round(4).to_csv(OUTPUT_DIR / "part_c_tuning.csv", index=False)

# Task 5: highest validation F1; dict order is the tie-break order (default, then grid, then randomized).
params_by_search = {"default_forest_same_7_features": DEFAULT_PARAMS,
                    "grid_search": grid_search.best_params_,
                    "randomized_search": random_search.best_params_}
val_f1 = {row["search"]: row["val_f1"] for row in tuning_rows}
chosen_from = next(name for name in params_by_search if val_f1[name] == max(val_f1.values()))
chosen = params_by_search[chosen_from]
final_model = Pipeline([("selector", SelectKBest(f_classif, k=SELECTOR_K)),
                        ("model", RandomForestClassifier(random_state=42, **chosen))]).fit(X_train, y_train)

# Task 6: highest threshold that still catches at least 80% of delayed parcels on validation.
val_proba = final_model.predict_proba(X_val)[:, 1]
threshold_rows = [{"threshold": t, **scorecard(y_val, (val_proba >= t).astype(int))} for t in THRESHOLDS]
pd.DataFrame(threshold_rows).round(4).to_csv(OUTPUT_DIR / "part_c_thresholds.csv", index=False)
eligible = [row["threshold"] for row in threshold_rows if row["recall"] >= MIN_RECALL]
alert_threshold = max(eligible) if eligible else 0.1

# Task 7: the single test evaluation.
test_pred = (final_model.predict_proba(X_test)[:, 1] >= alert_threshold).astype(int)
pd.DataFrame([{"threshold": alert_threshold, **scorecard(y_test, test_pred)}]).round(4).to_csv(
    OUTPUT_DIR / "part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": SELECTOR_K, "n_estimators": chosen["n_estimators"],
               "max_depth": "unlimited" if chosen["max_depth"] is None else chosen["max_depth"],
               "min_samples_leaf": chosen["min_samples_leaf"], "threshold": alert_threshold,
               "chosen_from": chosen_from}]).to_csv(OUTPUT_DIR / "part_c_final_config.csv", index=False)

print(f"best tree depth {best_depth}; final config from {chosen_from} {chosen}; alert threshold {alert_threshold}")
