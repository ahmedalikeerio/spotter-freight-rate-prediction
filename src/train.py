
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline

from src.preprocessing import build_features
from src.modeling import make_preprocessor, make_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def train_model(df, model_name, use_geography):

    # Build the required input features
    X = build_features(
        df,
        include_optional=use_geography
    )

    if use_geography:
        X = X.drop(
            columns=["market_index", "quote_signal"]
        )

    y = df["posted_rate"]

    # Create the chosen LightGBM configuration
    estimator = make_models()["LightGBM"]

    estimator.set_params(
        num_leaves=45,
        min_child_samples=40,
        reg_lambda=3.0
    )

    # Combine preprocessing and regression
    pipeline = Pipeline([
        ("preprocessing", make_preprocessor(X)),
        ("model", estimator)
    ])

    # Fit on all available labeled development data
    pipeline.fit(X, y)

    # Save the fitted pipeline and its feature schema
    output_path = MODEL_DIR / f"{model_name}.joblib"

    dump(
        {
            "pipeline": pipeline,
            "feature_columns": X.columns.tolist()
        },
        output_path
    )

    print(f"Saved: {output_path}")


def main():

    df = pd.read_csv(
        DATA_DIR / "train-test.csv"
    )

    print("Training records:", len(df))

    # Model for 12,000 validation loads
    train_model(
        df,
        model_name="freight_geography",
        use_geography=True
    )

    # Model for the December chart
    train_model(
        df,
        model_name="freight_common",
        use_geography=False
    )


if __name__ == "__main__":
    main()
