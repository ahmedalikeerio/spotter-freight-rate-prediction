
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def make_preprocessor(X):

    categorical_cols = [
        "pickup",
        "delivery",
        "equipment"
    ]

    numerical_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])


def make_models():

    return {
        "LightGBM": LGBMRegressor(
            objective="regression_l1",
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),

        "XGBoost": XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        ),

        "CatBoost": CatBoostRegressor(
            loss_function="MAE",
            iterations=350,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            thread_count=4,
            verbose=False
        )
    }
