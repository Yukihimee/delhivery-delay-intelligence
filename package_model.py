"""Part D: refit the final Part C classifier on the training rows and save it as reusable artefacts."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from src.shipment_prep import (DATA_DIR, FEATURES, MODEL_DIR, OUTPUT_DIR, clean_shipments,
                               three_way_split)

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
config = pd.read_csv(OUTPUT_DIR / "part_c_final_config.csv").iloc[0]
max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
threshold = float(config["threshold"])

table = pd.read_csv(DATA_DIR / "modelling_table.csv")
X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(table[FEATURES], table["is_delayed"],
                                                                 stratify=True)
model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42, n_estimators=int(config["n_estimators"]),
                                     max_depth=max_depth, min_samples_leaf=int(config["min_samples_leaf"]))),
]).fit(X_train, y_train)

joblib.dump(model, MODEL_DIR / "delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv(MODEL_DIR / "delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv(MODEL_DIR / "feature_order.csv", index=False)
_, medians = clean_shipments(pd.read_csv(DATA_DIR / "raw_shipments.csv"))
pd.DataFrame({"column": list(medians), "median": list(medians.values())}).round(4).to_csv(
    MODEL_DIR / "impute_values.csv", index=False)

reloaded = joblib.load(MODEL_DIR / "delay_model.pkl")
diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - model.predict_proba(X_test)[:, 1]).max()
pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
    "value": [len(X_test), round(float(diff), 4), threshold],
}, dtype=object).to_csv(OUTPUT_DIR / "part_d_reload_check.csv", index=False)

print(f"saved {MODEL_DIR / 'delay_model.pkl'}; reload max |diff| on {len(X_test)} test rows = {diff}")
