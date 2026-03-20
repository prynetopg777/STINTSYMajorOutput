import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    parser = argparse.ArgumentParser(description="Supervised MCO pipeline for MyAnimeList")
    parser.add_argument(
        "--input",
        default="filtered-dataset/master_df.csv",
        help="Path to master_df.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for model metrics and artifacts",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.15,
        help="Fraction of rows to keep via stratified sampling",
    )
    parser.add_argument(
        "--rf-iter",
        type=int,
        default=8,
        help="Iterations for RandomizedSearchCV on Random Forest",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Max epochs for PyTorch MLP",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for PyTorch MLP",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest", "pytorch_mlp"],
        default="logistic_regression",
        help="Run one model at a time. Start with logistic_regression.",
    )
    return parser.parse_args()


def load_master_df(path: str) -> pd.DataFrame:
    keep_columns = [
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
    missing_cols = [c for c in keep_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from input: {missing_cols}")

    df = df[keep_columns].copy()
    df["is_recommended"] = (df["my_score"] >= THRESHOLD).astype(int)
    return df


def stratified_row_sample(df: pd.DataFrame, sample_frac: float, seed: int) -> pd.DataFrame:
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")
    if sample_frac == 1:
        return df.copy()

    sampled_df, _ = train_test_split(
        df,
        train_size=sample_frac,
        random_state=seed,
        stratify=df["is_recommended"],
    )
    return sampled_df.reset_index(drop=True)


def _split_genre_cell(cell: object) -> List[str]:
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []

    # Support either pipe-separated or comma-separated genre strings.
    if "|" in text:
        parts = text.split("|")
    else:
        parts = text.split(",")
    return [p.strip() for p in parts if p.strip()]


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    work = df.copy()

    # Leakage-safe feature selection.
    y = work["is_recommended"].copy()
    work = work.drop(
        columns=["is_recommended", "my_score", "score_vs_user_mean", "username", "anime_id", "title"]
    )

    work["user_age"] = work["user_age"].fillna(work["user_age"].median())
    work["episodes"] = work["episodes"].fillna(work["episodes"].median())
    work["completion_ratio"] = work["completion_ratio"].fillna(work["completion_ratio"].median())
    work["stats_mean_score"] = work["stats_mean_score"].fillna(work["stats_mean_score"].median())

    work["gender"] = work["gender"].fillna("Unknown").astype(str)
    work["type"] = work["type"].fillna("Unknown").astype(str)
    work["source"] = work["source"].fillna("Unknown").astype(str)

    mlb = MultiLabelBinarizer()
    genres = work["genre"].apply(_split_genre_cell)
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(genres),
        columns=[f"genre__{c}" for c in mlb.classes_],
        index=work.index,
    )

    cat_encoded = pd.get_dummies(work[["gender", "type", "source"]], drop_first=False)
    numeric_df = work[["user_age", "episodes", "completion_ratio", "stats_mean_score"]].copy()

    x = pd.concat([numeric_df, cat_encoded, genre_encoded], axis=1)

    metadata = {
        "numeric_columns": numeric_df.columns.tolist(),
        "categorical_columns": cat_encoded.columns.tolist(),
        "genre_columns": genre_encoded.columns.tolist(),
    }
    return x, y, metadata


def make_splits(x: pd.DataFrame, y: pd.Series, seed: int) -> Splits:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    return Splits(x_train, x_val, x_test, y_train, y_val, y_test)


def scale_numeric_inplace(splits: Splits, numeric_columns: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    # Ensure dtype can hold scaled float values.
    splits.x_train.loc[:, numeric_columns] = splits.x_train[numeric_columns].astype(float)
    splits.x_val.loc[:, numeric_columns] = splits.x_val[numeric_columns].astype(float)
    splits.x_test.loc[:, numeric_columns] = splits.x_test[numeric_columns].astype(float)
    splits.x_train.loc[:, numeric_columns] = scaler.fit_transform(splits.x_train[numeric_columns])
    splits.x_val.loc[:, numeric_columns] = scaler.transform(splits.x_val[numeric_columns])
    splits.x_test.loc[:, numeric_columns] = scaler.transform(splits.x_test[numeric_columns])
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


def train_logistic_regression(splits: Splits, seed: int) -> Tuple[LogisticRegression, Dict[str, object]]:
    base = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )

    grid = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
    }

    search = GridSearchCV(
        estimator=base,
        param_grid=grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    search.fit(splits.x_train, splits.y_train)

    best_model = search.best_estimator_
    val_probs = best_model.predict_proba(splits.x_val)[:, 1]
    metrics = evaluate(splits.y_val, val_probs)
    metrics["best_params"] = search.best_params_
    return best_model, metrics


def train_random_forest(
    splits: Splits, seed: int, rf_iter: int
) -> Tuple[RandomForestClassifier, Dict[str, object]]:
    base = RandomForestClassifier(
        random_state=seed,
        class_weight="balanced",
        n_estimators=200,
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators": [150, 200, 300, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=rf_iter,
        scoring="f1_macro",
        cv=3,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(splits.x_train, splits.y_train)

    best_model = search.best_estimator_
    val_probs = best_model.predict_proba(splits.x_val)[:, 1]
    metrics = evaluate(splits.y_val, val_probs)
    metrics["best_params"] = search.best_params_
    return best_model, metrics


if TORCH_AVAILABLE:

    class AnimeDataset(Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, idx: int):
            return self.x[idx], self.y[idx]


    class MLPClassifier(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def train_pytorch_mlp(
    splits: Splits,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[Optional[object], Dict[str, object]]:
    if not TORCH_AVAILABLE:
        return None, {"skipped": "PyTorch is not installed in the current environment."}

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train = splits.x_train.to_numpy(dtype=np.float32)
    y_train = splits.y_train.to_numpy(dtype=np.float32)
    x_val = splits.x_val.to_numpy(dtype=np.float32)
    y_val = splits.y_val.to_numpy(dtype=np.float32)

    train_ds = AnimeDataset(x_train, y_train)
    val_ds = AnimeDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(input_dim=x_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state = None
    best_val_loss = float("inf")
    patience = 5
    no_improve = 0

    train_losses: List[float] = []
    val_losses: List[float] = []

    for _epoch in range(epochs):
        model.train()
        running_train = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_train += float(loss.item()) * len(xb)

        train_loss = running_train / len(train_ds)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                running_val += float(loss.item()) * len(xb)

        val_loss = running_val / len(val_ds)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_probs = model(torch.tensor(x_val, dtype=torch.float32)).squeeze(1).numpy()

    metrics = evaluate(splits.y_val, val_probs)
    metrics["epochs_ran"] = len(train_losses)
    metrics["train_loss_curve"] = train_losses
    metrics["val_loss_curve"] = val_losses
    return model, metrics


def evaluate_on_test(
    model_name: str,
    model: object,
    splits: Splits,
) -> Dict[str, object]:
    if model_name in {"logistic_regression", "random_forest"}:
        probs = model.predict_proba(splits.x_test)[:, 1]
        return evaluate(splits.y_test, probs)

    if model_name == "pytorch_mlp" and TORCH_AVAILABLE:
        x_test = torch.tensor(splits.x_test.to_numpy(dtype=np.float32), dtype=torch.float32)
        with torch.no_grad():
            probs = model(x_test).squeeze(1).numpy()
        return evaluate(splits.y_test, probs)

    raise ValueError(f"Unsupported model_name: {model_name}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    df = load_master_df(args.input)
    df_sampled = stratified_row_sample(df, args.sample_frac, RANDOM_STATE)
    print(f"Sampled rows: {len(df_sampled):,} / {len(df):,}")
    print("Class balance (sampled):")
    print(df_sampled["is_recommended"].value_counts(normalize=True).rename("ratio").to_string())

    x, y, metadata = preprocess_features(df_sampled)
    splits = make_splits(x, y, RANDOM_STATE)
    scaler = scale_numeric_inplace(splits, metadata["numeric_columns"])

    results: Dict[str, Dict[str, object]] = {}

    print(f"\nSelected model: {args.model}")
    if args.model == "logistic_regression":
        print("Training Logistic Regression with GridSearchCV...")
        model, val_metrics = train_logistic_regression(splits, RANDOM_STATE)
        results["logistic_regression_val"] = val_metrics
        test_metrics = evaluate_on_test("logistic_regression", model, splits)
        results["logistic_regression_test"] = test_metrics
        val_summary = pd.DataFrame(
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
    elif args.model == "random_forest":
        print("Training Random Forest with RandomizedSearchCV...")
        model, val_metrics = train_random_forest(splits, RANDOM_STATE, args.rf_iter)
        results["random_forest_val"] = val_metrics
        test_metrics = evaluate_on_test("random_forest", model, splits)
        results["random_forest_test"] = test_metrics
        val_summary = pd.DataFrame(
            [
                {
                    "model": "random_forest",
                    "f1_macro": val_metrics["f1_macro"],
                    "roc_auc": val_metrics["roc_auc"],
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                }
            ]
        )
    else:
        print("Training PyTorch MLP...")
        model, val_metrics = train_pytorch_mlp(
            splits,
            RANDOM_STATE,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results["pytorch_mlp_val"] = val_metrics
        if model is None:
            raise RuntimeError("PyTorch model selected but PyTorch is not installed.")

        test_metrics = evaluate_on_test("pytorch_mlp", model, splits)
        results["pytorch_mlp_test"] = test_metrics
        val_summary = pd.DataFrame(
            [
                {
                    "model": "pytorch_mlp",
                    "f1_macro": val_metrics["f1_macro"],
                    "roc_auc": val_metrics["roc_auc"],
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                }
            ]
        )

    artifacts = {
        "config": {
            "input": args.input,
            "sample_frac": args.sample_frac,
            "random_state": RANDOM_STATE,
            "threshold": THRESHOLD,
            "rows_total": int(len(df)),
            "rows_sampled": int(len(df_sampled)),
        },
        "feature_metadata": metadata,
        "results": results,
    }

    metrics_json_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    val_csv_path = os.path.join(args.output_dir, "validation_comparison.csv")
    val_summary.to_csv(val_csv_path, index=False)

    # Persist scaler feature order for reproducibility.
    scaler_cols_path = os.path.join(args.output_dir, "scaled_numeric_columns.json")
    with open(scaler_cols_path, "w", encoding="utf-8") as f:
        json.dump({"columns": metadata["numeric_columns"]}, f, indent=2)

    print("\nSaved artifacts:")
    print(f"- {metrics_json_path}")
    print(f"- {val_csv_path}")
    print(f"- {scaler_cols_path}")


if __name__ == "__main__":
    main()
