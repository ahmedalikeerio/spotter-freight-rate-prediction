
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def predict_rates(data, model_name, use_geography):

    # Load saved model and preprocessing pipeline
    artifact = joblib.load(
        MODEL_DIR / f"{model_name}.joblib"
    )

    pipeline = artifact["pipeline"]
    expected_columns = artifact["feature_columns"]

    # Create the same features used during training
    X = build_features(
        data,
        include_optional=use_geography
    )

    if use_geography:
        X = X.drop(
            columns=["market_index", "quote_signal"]
        )

    # Check feature compatibility
    if set(X.columns) != set(expected_columns):
        raise ValueError("Feature schema mismatch")

    X = X[expected_columns]

    predictions = pipeline.predict(X)

    # Validate numerical predictions
    if (
        not np.isfinite(predictions).all()
        or (predictions <= 0).any()
    ):
        raise ValueError("Invalid predicted freight rates")

    return predictions


def main():

    # Part 1: Predict 12,000 validation loads
    validation = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    template = pd.read_csv(
        DATA_DIR / "validation-predictions-template.csv"
    )

    assert len(validation) == 12000
    assert len(template) == 12000
    assert validation["load_id"].is_unique
    assert template["load_id"].is_unique

    validation_predictions = predict_rates(
        validation,
        model_name="freight_geography",
        use_geography=True
    )

    # Match predictions using load_id, not row position
    lookup = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": validation_predictions
    })

    submission = template[["load_id"]].merge(
        lookup,
        on="load_id",
        how="left",
        validate="one_to_one",
        sort=False
    )

    assert submission["predicted_rate"].notna().all()

    submission.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False
    )

    print("Saved 12,000 validation predictions.")

    # Part 2: Predict all 31 December dates
    december = pd.read_csv(
        DATA_DIR / "december-chart-inputs.csv"
    )

    assert len(december) == 31

    december["predicted_rate"] = predict_rates(
        december,
        model_name="freight_common",
        use_geography=False
    )

    december.to_csv(
        OUTPUT_DIR / "december_chart_inputs.csv",
        index=False
    )

    print("Saved 31 December predictions.")


if __name__ == "__main__":
    main()
