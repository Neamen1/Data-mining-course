from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from models import CollaborativeFilteringModel, MatrixFactorizationModel
from preprocessing import preprocess_all


def _popular_items(ratings_train: pd.DataFrame) -> List[int]:
    return (
        ratings_train.groupby("movieId")["rating"]
        .mean()
        .sort_values(ascending=False)
        .index.astype(int)
        .tolist()
    )

# We generate recommendations for the test users using the specified model, and ensure that we have k recommendations for each user by filling in with popular items if necessary. We also validate that the output does not contain missing values or invalid movie IDs.
def _ensure_top_k_recommendations(
    recs: pd.DataFrame,
    user_ids: List[int],
    seen_by_user: Dict[int, set[int]],
    candidate_items: List[int],
    k: int,
) -> pd.DataFrame:
    rows = []

    recs_by_user = {
        int(uid): grp.sort_values("rank")["item_id"].astype(int).tolist()
        for uid, grp in recs.groupby("user_id")
    }

    for user_id in user_ids:
        picked = []
        existing = recs_by_user.get(int(user_id), [])

        for movie_id in existing:
            if movie_id not in picked:
                picked.append(movie_id)
            if len(picked) == k:
                break

        seen = seen_by_user.get(int(user_id), set())
        for movie_id in candidate_items:
            if len(picked) == k:
                break
            if movie_id in picked or movie_id in seen:
                continue
            picked.append(movie_id)

        if len(picked) < k:
            raise RuntimeError(f"Could not produce {k} recommendations for user {user_id}")

        row = {"userId": int(user_id)}
        for idx in range(1, k + 1):
            row[f"recommendation{idx}"] = int(picked[idx - 1])
        rows.append(row)

    return pd.DataFrame(rows)


def generate_recommendations(
    data_dir: str | Path = ".",
    output_path: str | Path = "ratings_test_completed.csv",
    k: int = 10,
    model_code: str = "mf",
) -> pd.DataFrame:
    data = preprocess_all(data_dir)
    # We select the model based on the provided code
    normalized = model_code.strip().lower()
    if normalized == "mf":
        model = MatrixFactorizationModel()
    elif normalized == "cf":
        model = CollaborativeFilteringModel()
    else:
        raise ValueError(f"Unsupported model code: {model_code}. Use one of: cf, mf")
    
    model.train(train_df=data.ratings_train, movies_df=data.movies)

    test_users = data.ratings_test_users["userId"].astype(int).tolist()
    preds = model.predict(test_users, k=k)
    # We ensure that we have k recommendations for each user by filling in with popular items if necessary, while avoiding items the user has already seen
    popularity_fallback = _popular_items(data.ratings_train)
    completed = _ensure_top_k_recommendations(
        recs=preds,
        user_ids=test_users,
        seen_by_user=model.user_seen_items,
        candidate_items=popularity_fallback,
        k=k,
    )

    valid_movie_ids = set(data.movies["movieId"].astype(int).tolist())
    recommendation_cols = [f"recommendation{i}" for i in range(1, k + 1)]

    if completed[recommendation_cols].isna().sum().sum() != 0:
        raise RuntimeError("Output contains missing recommendations")

    invalid_count = (~completed[recommendation_cols].isin(valid_movie_ids)).sum().sum()
    if int(invalid_count) != 0:
        raise RuntimeError("Output contains invalid movieIds")

    completed.to_csv(output_path, index=False)
    return completed


if __name__ == "__main__":
    output = generate_recommendations(".", "ratings_test_completed.csv", k=10, model_code="mf")
    print("Generated recommendations:")
    print(output.head().to_string(index=False))
    print(f"Rows: {len(output)}")
