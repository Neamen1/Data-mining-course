from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from metrics import calculate_item_coverage, calculate_item_gini, calculate_ndcg
from models import CollaborativeFilteringModel, MatrixFactorizationModel
from pipeline import RecommenderModel, RecommenderPipeline
from preprocessing import preprocess_all


@dataclass
class HoldoutSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


def temporal_user_holdout_split(
    ratings_df: pd.DataFrame,
    min_user_ratings: int = 5,
    validation_ratio: float = 0.2,
) -> HoldoutSplit:
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    # We group by user and sort their ratings by timestamp to create a temporal split
    for _, group in ratings_df.groupby("userId"):
        group_sorted = group.sort_values("timestamp")
        if len(group_sorted) < min_user_ratings:
            train_parts.append(group_sorted)
            continue

        val_size = max(1, int(len(group_sorted) * validation_ratio))
        val_parts.append(group_sorted.tail(val_size))
        train_parts.append(group_sorted.iloc[:-val_size])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=ratings_df.columns)
    return HoldoutSplit(train=train_df, validation=val_df)


def _build_relevance_map(ground_truth: pd.DataFrame) -> Dict[int, set[int]]:
    relevance: Dict[int, set[int]] = {}
    for user_id, group in ground_truth.groupby("userId"):
        relevance[int(user_id)] = set(group["movieId"].astype(int).tolist())
    return relevance


def precision_recall_at_k(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int,
) -> Dict[str, float]:
    if ground_truth.empty:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}

    if predictions.empty:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}
    # We assume predictions have columns: user_id, item_id, score
    pred_topk = (
        predictions.sort_values(["user_id", "rank", "score"], ascending=[True, True, False])
        .groupby("user_id")
        .head(k)
    )

    relevance_map = _build_relevance_map(ground_truth)
    users = sorted(relevance_map.keys())

    precisions: List[float] = []
    recalls: List[float] = []
    # We compute precision and recall for each user and then average them
    for user_id in users:
        rel_items = relevance_map[user_id]
        if not rel_items:
            continue

        rec_items = pred_topk.loc[pred_topk["user_id"] == user_id, "item_id"].astype(int).tolist()
        hits = len(set(rec_items).intersection(rel_items))

        precisions.append(hits / float(k))
        recalls.append(hits / float(len(rel_items)))

    if not precisions:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
    }


def evaluate_model(
    model: RecommenderModel,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    k: int = 10,
    positive_threshold: float = 4.0,
) -> Dict[str, float]:
    # We train the model on the training data and then evaluate it on the validation set using RMSE and ranking metrics
    model.train(train_df=train_df, movies_df=movies_df)

    rmse_result = model.evaluate(validation_df, k=k)
    rmse_value = float(rmse_result.get("rmse", np.nan))

    ranking_truth = validation_df[validation_df["rating"] >= positive_threshold][["userId", "movieId"]].copy()
    eval_users = ranking_truth["userId"].drop_duplicates().astype(int).tolist()

    predictions = model.predict(eval_users, k=k)
    ranking_metrics = precision_recall_at_k(predictions, ranking_truth, k=k)

    predictions_for_ndcg = predictions[["user_id", "item_id"]].copy()
    ndcg_truth = ranking_truth.rename(columns={"userId": "user_id", "movieId": "item_id"})

    ndcg_value = 0.0
    if not ndcg_truth.empty and not predictions_for_ndcg.empty:
        ndcg_value = float(calculate_ndcg(predictions_for_ndcg, k, ndcg_truth))

    diversity_predictions = predictions[["user_id", "item_id"]].copy()
    item_coverage = float(calculate_item_coverage(diversity_predictions, k, movies_df["movieId"].nunique()))
    item_gini = float(calculate_item_gini(diversity_predictions, k))

    return {
        "rmse": rmse_value,
        "precision_at_k": ranking_metrics["precision_at_k"],
        "recall_at_k": ranking_metrics["recall_at_k"],
        "ndcg_at_k": ndcg_value,
        "item_coverage": item_coverage,
        "item_gini": item_gini,
    }


def run_eval_evaluation(k: int = 10) -> pd.DataFrame:
    return run_evaluation(data_dir=".", model_codes=["cf", "mf"], k=k)


def _build_models(model_codes: Iterable[str]) -> Dict[str, RecommenderModel]:
    model_map: Dict[str, RecommenderModel] = {}
    for code in model_codes:
        normalized = code.strip().lower()
        if normalized == "cf":
            model = CollaborativeFilteringModel()
        elif normalized == "mf":
            model = MatrixFactorizationModel()
        else:
            raise ValueError(f"Unsupported model code: {code}. Use one of: cf, mf")
        model_map[model.name] = model
    return model_map


def run_evaluation(
    data_dir: str = ".",
    model_codes: Iterable[str] = ("cf", "mf"),
    k: int = 10,
) -> pd.DataFrame:
    # We load and preprocess the data, create a temporal holdout split, build the specified models, evaluate each model, and return a DataFrame with the results
    preprocessed = preprocess_all(data_dir)
    split = temporal_user_holdout_split(preprocessed.ratings_train)

    pipeline = RecommenderPipeline()
    for model in _build_models(model_codes).values():
        pipeline.register_model(model)

    rows: List[Dict[str, float]] = []
    for model_name, model in pipeline.models.items():
        metrics = evaluate_model(
            model=model,
            train_df=split.train,
            validation_df=split.validation,
            movies_df=preprocessed.movies,
            k=k,
        )
        rows.append({"model": model_name, **metrics})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_eval_evaluation(k=10)
    print("Eval evaluation results:")
    print(results.to_string(index=False))
