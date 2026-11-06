"""Shared constants and preprocessing used by every Part and by predict_delay.py."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
HUB_COLUMNS = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
# The network accepts parcels up to 40 kg, so a weight above 100 kg is a scanner-fault sentinel.
WEIGHT_SENTINEL_KG = 100
SPLIT_SEED = 42


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export and return (clean_df, medians); pass stored medians for new parcels."""
    df = raw.drop_duplicates().reset_index(drop=True)
    df["service_type"] = df["service_type"].str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    value_text = df["declared_value"].str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(value_text)
    df = df.drop(columns="declared_value")
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M")
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_KG, "parcel_weight_kg"] = np.nan
    if medians is None:
        medians = df[MEDIAN_COLUMNS].median()
    medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    df[MEDIAN_COLUMNS] = df[MEDIAN_COLUMNS].fillna(medians)
    return df, medians


def build_features(clean, hubs):
    """Attach the hub columns and derive the model inputs; never reads delivery_hours."""
    df = clean.merge(hubs[HUB_COLUMNS], on="hub_id", how="left", validate="many_to_one")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df


def three_way_split(X, y, stratify=False):
    """70 / 15 / 15 train / validation / test split made with two train_test_split calls and the pinned seed."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SPLIT_SEED, stratify=y if stratify else None)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SPLIT_SEED, stratify=y_temp if stratify else None)
    return X_train, X_val, X_test, y_train, y_val, y_test
