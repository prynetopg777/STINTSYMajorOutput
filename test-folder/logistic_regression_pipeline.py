import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

RANDOM_STATE = 42
THRESHOLD = 8


@dataclass
class Splits:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Logistic Regression pipeline for MCO"
    )
    parser.add_argument(
        "--input",
        default="filtered-dataset/master_df.csv",
        help="Path to master_df.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_logistic_regression",
        help="Directory for metrics and artifacts",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.15,
        help="Fraction of rows to keep using stratified sampling",
    )
    return parser.parse_args()


def load_master_df(path: str) -> pd.DataFrame:
    required_columns = [
        "username",
        "anime_id",
        "title",
        "my_score",
        "gender",
        "user_age",
        "stats_mean_score",
        "type",
        "episodes",
        "source",
        "genre",
        "completion_ratio",
        "score_vs_user_mean",
    ]

    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_columns].copy()
    df["is_recommended"] = (df["my_score"] >= THRESHOLD).astype(int)
    return df


def stratified_sample(df: pd.DataFrame, sample_frac: float) -> pd.DataFrame:
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")
    if sample_frac == 1:
        return df.copy()

    sampled, _ = train_test_split(
        df,
        train_size=sample_frac,
        random_state=RANDOM_STATE,
        stratify=df["is_recommended"],
    )
    return sampled.reset_index(drop=True)


def split_genres(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []

    delimiter = "|" if "|" in text else ","
    return [token.strip() for token in text.split(delimiter) if token.strip()]


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    y = df["is_recommended"].copy()

    x_raw = df.drop(
        columns=[
            "is_recommended",
            "my_score",
            "score_vs_user_mean",
            "username",
            "anime_id",
            "title",
        ]
    ).copy()

    x_raw["user_age"] = x_raw["user_age"].fillna(x_raw["user_age"].median())
    x_raw["episodes"] = x_raw["episodes"].fillna(x_raw["episodes"].median())
    x_raw["completion_ratio"] = x_raw["completion_ratio"].fillna(
        x_raw["completion_ratio"].median()
    )
    x_raw["stats_mean_score"] = x_raw["stats_mean_score"].fillna(
        x_raw["stats_mean_score"].median()
    )

    x_raw["gender"] = x_raw["gender"].fillna("Unknown").astype(str)
    x_raw["type"] = x_raw["type"].fillna("Unknown").astype(str)
    x_raw["source"] = x_raw["source"].fillna("Unknown").astype(str)

    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(x_raw["genre"].apply(split_genres)),
        columns=[f"genre__{genre}" for genre in mlb.classes_],
        index=x_raw.index,
    )

    categorical = pd.get_dummies(x_raw[["gender", "type", "source"]], drop_first=False)
    numeric = x_raw[["user_age", "episodes", "completion_ratio", "stats_mean_score"]].copy()

    x = pd.concat([numeric, categorical, genre_encoded], axis=1)

    metadata = {
        "numeric_columns": numeric.columns.tolist(),
        "categorical_columns": categorical.columns.tolist(),
        "genre_columns": genre_encoded.columns.tolist(),
    }
    return x, y, metadata


def make_splits(x: pd.DataFrame, y: pd.Series) -> Splits:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return Splits(x_train, x_val, x_test, y_train, y_val, y_test)


def scale_numeric(splits: Splits, numeric_columns: List[str]) -> StandardScaler:
    dtype_map = {col: "float64" for col in numeric_columns}
    splits.x_train = splits.x_train.astype(dtype_map)
    splits.x_val = splits.x_val.astype(dtype_map)
    splits.x_test = splits.x_test.astype(dtype_map)

    scaler = StandardScaler()
    splits.x_train[numeric_columns] = scaler.fit_transform(
        splits.x_train[numeric_columns].to_numpy(dtype=np.float64)
    )
    splits.x_val[numeric_columns] = scaler.transform(
        splits.x_val[numeric_columns].to_numpy(dtype=np.float64)
    )
    splits.x_test[numeric_columns] = scaler.transform(
        splits.x_test[numeric_columns].to_numpy(dtype=np.float64)
    )
    return scaler


def evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_logistic_gridsearch(splits: Splits) -> Tuple[LogisticRegression, Dict[str, object]]:
    base = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )

    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
    }

    grid_search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    grid_search.fit(splits.x_train, splits.y_train)

    model = grid_search.best_estimator_
    val_probs = model.predict_proba(splits.x_val)[:, 1]
    val_metrics = evaluate(splits.y_val, val_probs)
    val_metrics["best_params"] = grid_search.best_params_

    return model, val_metrics


def evaluate_on_test(model: LogisticRegression, splits: Splits) -> Dict[str, object]:
    test_probs = model.predict_proba(splits.x_test)[:, 1]
    return evaluate(splits.y_test, test_probs)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading and sampling data...")
    df = load_master_df(args.input)
    sampled_df = stratified_sample(df, args.sample_frac)

    print(f"Rows: {len(sampled_df):,} / {len(df):,}")
    print("Class ratio in sample:")
    print(sampled_df["is_recommended"].value_counts(normalize=True).to_string())

    x, y, metadata = preprocess(sampled_df)
    splits = make_splits(x, y)
    scale_numeric(splits, metadata["numeric_columns"])

    print("Training Logistic Regression with GridSearchCV...")
    model, val_metrics = train_logistic_gridsearch(splits)

    print("Evaluating on held-out test set...")
    test_metrics = evaluate_on_test(model, splits)

    artifacts = {
        "config": {
            "input": args.input,
            "sample_frac": args.sample_frac,
            "random_state": RANDOM_STATE,
            "threshold": THRESHOLD,
            "rows_total": int(len(df)),
            "rows_sampled": int(len(sampled_df)),
        },
        "feature_metadata": metadata,
        "results": {
            "logistic_regression_val": val_metrics,
            "logistic_regression_test": test_metrics,
        },
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    summary_df = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "f1_macro": val_metrics["f1_macro"],
                "roc_auc": val_metrics["roc_auc"],
                "accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
            }
        ]
    )
    summary_path = os.path.join(args.output_dir, "validation_comparison.csv")
    summary_df.to_csv(summary_path, index=False)

    numeric_cols_path = os.path.join(args.output_dir, "scaled_numeric_columns.json")
    with open(numeric_cols_path, "w", encoding="utf-8") as f:
        json.dump({"columns": metadata["numeric_columns"]}, f, indent=2)

    print("Saved artifacts:")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")
    print(f"- {numeric_cols_path}")


if __name__ == "__main__":
    main()
