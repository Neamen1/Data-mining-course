from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass
class PreprocessedData:
    movies: pd.DataFrame
    ratings_train: pd.DataFrame
    ratings_test_users: pd.DataFrame
    ratings_with_movies: pd.DataFrame
    genre_columns: List[str]


def load_raw_data(data_dir: str | Path = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings_train = pd.read_csv(data_dir / "ratings_train.csv")
    ratings_test = pd.read_csv(data_dir / "ratings_test.csv")
    return movies, ratings_train, ratings_test


def inspect_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
    }


def preprocess_movies(movies: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cleaned = movies.copy()

    cleaned["title"] = cleaned["title"].fillna("Unknown Title")
    cleaned["genres"] = cleaned["genres"].fillna("Unknown")

    genre_lists = cleaned["genres"].astype(str).str.split("|")
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genre_lists)
    genre_columns = [f"genre_{genre}" for genre in mlb.classes_]

    genre_df = pd.DataFrame(genre_matrix, columns=genre_columns, index=cleaned.index)
    cleaned = pd.concat([cleaned, genre_df], axis=1)

    return cleaned, genre_columns


def preprocess_ratings(ratings_train: pd.DataFrame) -> pd.DataFrame:
    cleaned = ratings_train.copy()

    required_cols = ["userId", "movieId", "rating", "timestamp"]
    missing_cols = [col for col in required_cols if col not in cleaned.columns]
    if missing_cols:
        raise ValueError(f"ratings_train is missing required columns: {missing_cols}")

    cleaned = cleaned.dropna(subset=["userId", "movieId", "rating"])
    cleaned["userId"] = cleaned["userId"].astype(int)
    cleaned["movieId"] = cleaned["movieId"].astype(int)
    cleaned["rating"] = cleaned["rating"].astype(float)

    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_numeric(cleaned["timestamp"], errors="coerce")
        cleaned["timestamp"] = cleaned["timestamp"].fillna(0).astype(int)

    # Keep ratings in the common explicit-feedback range for stability.
    cleaned["rating"] = cleaned["rating"].clip(lower=0.5, upper=5.0)
    return cleaned


def preprocess_test_users(ratings_test: pd.DataFrame) -> pd.DataFrame:
    if "userId" not in ratings_test.columns:
        raise ValueError("ratings_test must contain a userId column")

    cleaned = ratings_test[["userId"]].dropna().copy()
    cleaned["userId"] = cleaned["userId"].astype(int)
    cleaned = cleaned.drop_duplicates(subset=["userId"]).reset_index(drop=True)
    return cleaned


def build_merged_dataset(
    ratings_train: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    merged = ratings_train.merge(movies, on="movieId", how="left", validate="many_to_one")
    merged["title"] = merged["title"].fillna("Unknown Title")
    merged["genres"] = merged["genres"].fillna("Unknown")
    return merged


def preprocess_all(data_dir: str | Path = ".") -> PreprocessedData:
    movies_raw, ratings_train_raw, ratings_test_raw = load_raw_data(data_dir)

    movies_clean, genre_columns = preprocess_movies(movies_raw)
    ratings_train_clean = preprocess_ratings(ratings_train_raw)
    ratings_test_users = preprocess_test_users(ratings_test_raw)
    ratings_with_movies = build_merged_dataset(ratings_train_clean, movies_clean)

    return PreprocessedData(
        movies=movies_clean,
        ratings_train=ratings_train_clean,
        ratings_test_users=ratings_test_users,
        ratings_with_movies=ratings_with_movies,
        genre_columns=genre_columns,
    )


def validate_preprocessed_data(preprocessed: PreprocessedData) -> Dict[str, bool]:
    checks = {
        "ratings_missing_handled": preprocessed.ratings_train[["userId", "movieId", "rating"]]
        .isna()
        .sum()
        .sum()
        == 0,
        "categorical_encoded": len(preprocessed.genre_columns) > 0,
        "merged_has_movie_metadata": {
            "title", "genres"
        }.issubset(set(preprocessed.ratings_with_movies.columns)),
    }
    return checks


def run_preprocessing_validation(data_dir: str | Path = ".") -> Dict[str, object]:
    movies, ratings_train, ratings_test = load_raw_data(data_dir)
    preprocessed = preprocess_all(data_dir)
    checks = validate_preprocessed_data(preprocessed)

    return {
        "movies_schema": inspect_dataframe(movies),
        "ratings_train_schema": inspect_dataframe(ratings_train),
        "ratings_test_schema": inspect_dataframe(ratings_test),
        "preprocessed_shapes": {
            "movies": preprocessed.movies.shape,
            "ratings_train": preprocessed.ratings_train.shape,
            "ratings_test_users": preprocessed.ratings_test_users.shape,
            "ratings_with_movies": preprocessed.ratings_with_movies.shape,
        },
        "validation_checks": checks,
        "all_checks_pass": all(checks.values()),
    }


if __name__ == "__main__":
    result = run_preprocessing_validation(".")
    print("Preprocessing validation summary:")
    print(result)
