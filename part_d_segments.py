"""Part D: profile the 24 hubs, cluster them into three operating segments and check the tier labels."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import DATA_DIR, OUTPUT_DIR

PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]  # clusters ordered by rising mean delay rate
TIER_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}
N_SEGMENTS = 3  # fixed by the business: operations runs exactly three playbooks

OUTPUT_DIR.mkdir(exist_ok=True)
table = pd.read_csv(DATA_DIR / "modelling_table.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

profile = (table.groupby("hub_id")
           .agg(shipments=("shipment_id", "size"), mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .sort_index()
           .reset_index())
profile.round(4).to_csv(OUTPUT_DIR / "hub_profile.csv", index=False)

# Hubs enter the scaler and K-Means in ascending hub_id order (HUB01 first).
scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
profile["cluster"] = KMeans(n_clusters=N_SEGMENTS, random_state=42, n_init=10).fit_predict(scaled)

pair_a, pair_b = np.triu_indices(len(scaled), k=1)  # the 276 unordered hub pairs
pair_distance = pairwise_distances(scaled)[pair_a, pair_b]
separation_rows = []
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    same = labels[pair_a] == labels[pair_b]
    separation_rows.append({
        "K": k,
        "across_over_within": pair_distance[~same].mean() / pair_distance[same].mean(),
        "cluster_sizes": "-".join(str(size) for size in sorted(np.bincount(labels), reverse=True)),
    })
pd.DataFrame(separation_rows).round(4).to_csv(OUTPUT_DIR / "part_d_separation.csv", index=False)

cluster_order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
profile["segment"] = profile["cluster"].map(dict(zip(cluster_order, SEGMENT_NAMES)))

segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments.to_csv(OUTPUT_DIR / "hub_segments.csv", index=False)

segment_profile = (profile.groupby("segment")
                   .agg(hubs=("hub_id", "size"), **{col: (col, "mean") for col in PROFILE_COLUMNS})
                   .reindex(SEGMENT_NAMES)
                   .rename_axis("segment")
                   .reset_index())
segment_profile.round(4).to_csv(OUTPUT_DIR / "segment_profile.csv", index=False)

mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_SEGMENT)]
mismatch.to_csv(OUTPUT_DIR / "tier_vs_segment_mismatch.csv", index=False)

print(f"segments: {segment_profile[['segment', 'hubs']].values.tolist()}; "
      f"tier mismatches: {mismatch['hub_id'].tolist()}")
