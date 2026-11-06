"""Score a batch of new parcels: delay probability, alert flag and the action for each parcel."""
import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import DATA_DIR, MODEL_DIR, OUTPUT_DIR, build_features, clean_shipments

new_raw = pd.read_csv(DATA_DIR / "new_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")
model = joblib.load(MODEL_DIR / "delay_model.pkl")
threshold = float(pd.read_csv(MODEL_DIR / "delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv(MODEL_DIR / "feature_order.csv")["feature"].tolist()
impute = pd.read_csv(MODEL_DIR / "impute_values.csv")
segments = pd.read_csv(OUTPUT_DIR / "hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new_raw, medians=dict(zip(impute["column"], impute["median"])))
features = build_features(clean, hubs)

actions = features[["shipment_id", "hub_id"]].merge(segments, on="hub_id", how="left", validate="many_to_one")
actions["delay_probability"] = model.predict_proba(features[feature_order])[:, 1].round(4)
actions["flagged"] = (actions["delay_probability"] >= threshold).astype(int)
flagged = actions["flagged"] == 1
actions["action"] = np.select([flagged & (actions["segment"] == "Escalate"), flagged],
                              ["Escalate to hub manager", "Proactive customer call"], default="No action")
actions.to_csv(OUTPUT_DIR / "new_shipment_actions.csv", index=False)

print(f"{len(actions)} parcels scored at threshold {threshold}: {actions['flagged'].sum()} flagged; "
      f"{actions['action'].value_counts().to_dict()}")
