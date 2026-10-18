
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
]


def build_features(
    raw: pd.DataFrame,
    include_optional: bool = False
) -> pd.DataFrame:

    missing_columns = (
        set(REQUIRED_COLUMNS) - set(raw.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    X = raw[REQUIRED_COLUMNS].copy()

    for col in ["pickup", "delivery", "equipment"]:
        X[col] = (
            X[col]
            .astype("string")
            .str.strip()
            .fillna("Unknown")
            .astype("object")
        )


    
    # Validate distance
    X["distance"] = pd.to_numeric(
        X["distance"],
        errors="coerce"
    )

    if (
        X["distance"].isna().any()
        or (X["distance"] <= 0).any()
    ):
        raise ValueError(
            "Distance must be present and positive."
        )

    # Inspect original weight values
    original_weight = pd.to_numeric(
        X["weight"],
        errors="coerce"
    )

    # Preserve information about data quality
    X["weight_missing"] = (
        original_weight.isna().astype("int8")
    )

    X["weight_invalid"] = (
        original_weight.le(0).astype("int8")
    )

    # Treat invalid weights as missing
    X["weight"] = original_weight.mask(
        original_weight <= 0
    )

    # Feature Engineering


    dates = pd.to_datetime(
        X.pop("date"),
        errors="coerce"
    )

    if dates.isna().any():
        raise ValueError(
            "Invalid or missing shipment dates."
        )

    # Calendar features
    X["month"] = dates.dt.month
    X["weekday"] = dates.dt.dayofweek

    X["is_weekend"] = (
        dates.dt.dayofweek >= 5
    ).astype("int8")

    # Time trend feature
    X["days_since_2025_01_01"] = (
        dates - pd.Timestamp("2025-01-01")
    ).dt.days



    # Numerical transformation
    X["log_distance"] = np.log1p(
        X["distance"]
    )

    # Optional numeric signals
    if include_optional:
        for col in OPTIONAL_NUMERIC_COLUMNS:
            if col in raw.columns:
                X[col] = pd.to_numeric(
                    raw[col],
                    errors="coerce"
                )
            else:
                X[col] = np.nan

    return X


