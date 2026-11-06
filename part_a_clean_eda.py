"""Part A: clean the raw export, write the modelling table and the first delay evidence."""
import pandas as pd

from src.shipment_prep import (DATA_DIR, FEATURES, OUTPUT_DIR, WEIGHT_SENTINEL_KG,
                               build_features, clean_shipments)

DISTANCE_BINS = [0, 200, 400, 600, 2000]
DISTANCE_LABELS = ["0-200", "201-400", "401-600", "601+"]

OUTPUT_DIR.mkdir(exist_ok=True)
raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv(DATA_DIR / "modelling_table.csv", index=False)

# "Before" counts are taken after removing duplicates and before any cleaning or filling.
dedup = raw.drop_duplicates()
hours = table["delivery_hours"]
q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": dedup["distance_km"].isna().sum(),
    "missing_parcel_weight_before": dedup["parcel_weight_kg"].isna().sum(),
    "sentinel_weight_rows": (dedup["parcel_weight_kg"] > WEIGHT_SENTINEL_KG).sum(),
    "missing_declared_value_before": dedup["declared_value"].isna().sum(),
    "missing_payment_mode_before": dedup["payment_mode"].isna().sum(),
    "service_type_spellings_before": dedup["service_type"].nunique(),
    "service_type_spellings_after": clean["service_type"].nunique(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(table),
    "nulls_in_features_and_target": table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum(),
    "delayed_count": table["is_delayed"].sum(),
    "delay_rate": table["is_delayed"].mean(),
    "hours_q1": q1,
    "hours_q3": q3,
    "hours_lower_bound": lower,
    "hours_upper_bound": upper,
    "hours_outlier_count": ((hours < lower) | (hours > upper)).sum(),
    "declared_value_mean_inr": table["declared_value_inr"].mean(),
    "declared_value_median_inr": table["declared_value_inr"].median(),
    "log_declared_value_mean": table["log_declared_value"].mean(),
    "log_declared_value_median": table["log_declared_value"].median(),
}
check_rows = [(name, round(float(v), 4) if isinstance(v, float) else int(v)) for name, v in checks.items()]
pd.DataFrame(check_rows, columns=["metric", "value"], dtype=object).to_csv(
    OUTPUT_DIR / "part_a_checks.csv", index=False)

dimensions = {
    "service_type": table["service_type"],
    "hub_tier": table["hub_tier"],
    "distance_band": pd.cut(table["distance_km"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS),
}
group_tables = []
for dimension, key in dimensions.items():
    summary = (table.groupby(key, observed=False)
               .agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                    delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
               .rename_axis("group")
               .reset_index())
    summary.insert(0, "dimension", dimension)
    group_tables.append(summary)
pd.concat(group_tables, ignore_index=True).round(4).to_csv(OUTPUT_DIR / "part_a_delay_by_group.csv", index=False)

table.select_dtypes("number").corr().round(4).to_csv(OUTPUT_DIR / "part_a_correlations.csv", index_label="column")

print(f"modelling_table.csv: {len(table)} rows, {checks['delayed_count']} delayed ({checks['delay_rate']:.4f})")
