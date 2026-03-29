from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

import pandas as pd
from surprise import Dataset, Reader, accuracy
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.matrix_factorization import SVD

from pipeline import RecommenderModel


@dataclass
class SurpriseArtifacts:
    trainset: object
    rating_scale: tuple[float, float]

# Base class for Surprise-based recommenders
class SurpriseRecommenderBase(RecommenderModel):
    def __init__(self, name: str, algorithm: AlgoBase) -> None:
        self._name = name
        self.algorithm = algorithm
        self.artifacts: SurpriseArtifacts | None = None
        self.user_seen_items: Dict[int, Set[int]] = {}
        self.candidate_items: List[int] = []
        self.global_mean: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    def train(self, train_df: pd.DataFrame, movies_df: pd.DataFrame | None = None) -> None:
        required = {"userId", "movieId", "rating"}
        if not required.issubset(set(train_df.columns)):
            raise ValueError(f"train_df must contain columns: {required}")

        # Prepare the data for Surprise
        ratings = train_df[["userId", "movieId", "rating"]].copy()
        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        
        # Surprise needs the rating scale to be defined, so we infer it from the data
        min_rating = float(ratings["rating"].min())
        max_rating = float(ratings["rating"].max())
        reader = Reader(rating_scale=(min_rating, max_rating))
        surprise_data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

        # Train the model
        trainset = surprise_data.build_full_trainset()
        self.algorithm.fit(trainset)

        self.user_seen_items = (
            ratings.groupby("userId")["movieId"].apply(lambda s: set(s.astype(int).tolist())).to_dict()
        )

        if movies_df is not None and "movieId" in movies_df.columns:
            self.candidate_items = sorted(movies_df["movieId"].dropna().astype(int).unique().tolist())
        else:
            self.candidate_items = sorted(ratings["movieId"].dropna().astype(int).unique().tolist())

        self.global_mean = float(ratings["rating"].mean())
        self.artifacts = SurpriseArtifacts(trainset=trainset, rating_scale=(min_rating, max_rating))

    def predict(self, user_ids: Iterable[int], k: int = 10) -> pd.DataFrame:
        self._ensure_trained()

        rows: List[Dict[str, float]] = []
        for user_id in user_ids:
            uid = int(user_id)
            seen = self.user_seen_items.get(uid, set())
            candidates = [iid for iid in self.candidate_items if iid not in seen]

            # For each candidate item, predict the rating for this user
            scored_items = []
            for item_id in candidates:
                est = self.algorithm.predict(uid=uid, iid=int(item_id)).est
                scored_items.append((int(item_id), float(est)))

            scored_items.sort(key=lambda x: x[1], reverse=True)
            top_items = scored_items[:k]

            for rank, (item_id, score) in enumerate(top_items, start=1):
                rows.append(
                    {
                        "user_id": uid,
                        "item_id": item_id,
                        "score": score,
                        "rank": rank,
                    }
                )

        return pd.DataFrame(rows, columns=["user_id", "item_id", "score", "rank"])

    def evaluate(self, eval_df: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        _ = k
        self._ensure_trained()

        required = {"userId", "movieId", "rating"}
        if not required.issubset(set(eval_df.columns)):
            return {}
        
        # Surprise's accuracy metrics expect a list of (user, item, true_rating) tuples
        testset = [
            (int(u), int(i), float(r))
            for u, i, r in eval_df[["userId", "movieId", "rating"]].itertuples(index=False)
        ]
        predictions = self.algorithm.test(testset)
        rmse_value = accuracy.rmse(predictions, verbose=False)
        return {"rmse": float(rmse_value)}

    def _ensure_trained(self) -> None:
        if self.artifacts is None:
            raise RuntimeError(f"Model '{self.name}' must be trained before use")


class CollaborativeFilteringModel(SurpriseRecommenderBase):
    def __init__(
        self,
        k_neighbors: int = 40,
        min_k: int = 3,
        sim_name: str = "cosine",
        user_based: bool = True,
    ) -> None:
        sim_options = {"name": sim_name, "user_based": user_based}
        algorithm = KNNBasic(k=k_neighbors, min_k=min_k, sim_options=sim_options, verbose=False)
        super().__init__(name="collaborative_filtering", algorithm=algorithm)


class MatrixFactorizationModel(SurpriseRecommenderBase):
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 100,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
    ) -> None:
        algorithm = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state,
            verbose=False,
        )
        super().__init__(name="matrix_factorization", algorithm=algorithm)


if __name__ == "__main__":
    from preprocessing import preprocess_all

    data = preprocess_all(".")
    train_df = data.ratings_train
    movies_df = data.movies

    cf_model = CollaborativeFilteringModel()
    mf_model = MatrixFactorizationModel()

    cf_model.train(train_df, movies_df)
    mf_model.train(train_df, movies_df)

    sample_users = train_df["userId"].drop_duplicates().head(3).tolist()
    cf_preds = cf_model.predict(sample_users, k=5)
    mf_preds = mf_model.predict(sample_users, k=5)

    # print("Models shape check:")
    # print({
    #     "cf_predictions": cf_preds.shape,
    #     "mf_predictions": mf_preds.shape,
    # })
