from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import pandas as pd

# This module defines the RecommenderModel interface and the RecommenderPipeline class to manage multiple models.
class RecommenderModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def train(self, train_df: pd.DataFrame, movies_df: pd.DataFrame | None = None) -> None:
        pass

    @abstractmethod
    def predict(self, user_ids: Iterable[int], k: int = 10) -> pd.DataFrame:
        pass

    @abstractmethod
    def evaluate(self, eval_df: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        pass

# The RecommenderPipeline class allows registering multiple models, training them, making predictions, and evaluating them in a consistent way.
class RecommenderPipeline:
    def __init__(self) -> None:
        self.models: Dict[str, RecommenderModel] = {}

    def register_model(self, model: RecommenderModel) -> None:
        if model.name in self.models:
            raise ValueError(f"Model '{model.name}' is already registered")
        self.models[model.name] = model

    def train_all(self, train_df: pd.DataFrame, movies_df: pd.DataFrame | None = None) -> None:
        for model in self.models.values():
            model.train(train_df=train_df, movies_df=movies_df)

    def predict_with_model(self, model_name: str, user_ids: Iterable[int], k: int = 10) -> pd.DataFrame:
        model = self._get_model(model_name)
        return model.predict(user_ids=user_ids, k=k)

    def evaluate_model(self, model_name: str, eval_df: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        model = self._get_model(model_name)
        return model.evaluate(eval_df=eval_df, k=k)

    def evaluate_all(self, eval_df: pd.DataFrame, k: int = 10) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for model_name, model in self.models.items():
            results[model_name] = model.evaluate(eval_df=eval_df, k=k)
        return results

    def _get_model(self, model_name: str) -> RecommenderModel:
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' is not registered")
        return self.models[model_name]

# dummy model to test the pipeline functionality
class DummyModel(RecommenderModel):
    def __init__(self) -> None:
        self._name = "dummy"
        self._trained = False

    @property
    def name(self) -> str:
        return self._name

    def train(self, train_df: pd.DataFrame, movies_df: pd.DataFrame | None = None) -> None:
        _ = train_df, movies_df
        self._trained = True

    def predict(self, user_ids: Iterable[int], k: int = 10) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("DummyModel must be trained before predict")

        rows: List[Dict[str, int]] = []
        for user_id in user_ids:
            for rank in range(1, k + 1):
                rows.append({"user_id": int(user_id), "item_id": rank, "score": float(k - rank + 1)})
        return pd.DataFrame(rows)

    def evaluate(self, eval_df: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        _ = eval_df, k
        if not self._trained:
            raise RuntimeError("DummyModel must be trained before evaluate")
        return {"status": 1.0}


def run_pipeline_validation() -> Dict[str, Any]:
    pipeline = RecommenderPipeline()
    dummy = DummyModel()
    pipeline.register_model(dummy)

    train_df = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 11, 10],
            "rating": [4.0, 3.5, 5.0],
            "timestamp": [1, 2, 3],
        }
    )

    pipeline.train_all(train_df)
    preds = pipeline.predict_with_model("dummy", user_ids=[1, 2], k=3)

    checks = {
        "model_registration": "dummy" in pipeline.models,
        "prediction_columns": {"user_id", "item_id", "score"}.issubset(set(preds.columns)),
        "prediction_count": len(preds) == 6,
        "consistent_interface": isinstance(pipeline.evaluate_model("dummy", train_df, k=3), dict),
    }

    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


if __name__ == "__main__":
    result = run_pipeline_validation()
    print("Pipeline validation summary:")
    print(result)
