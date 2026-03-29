from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from evaluation import run_evaluation
from generate_recommendations import generate_recommendations
from preprocessing import preprocess_all
from visualization import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full recommender system pipeline")
    parser.add_argument("--data_path", required=True, help="Path containing movies.csv, ratings_train.csv, ratings_test.csv")
    parser.add_argument("--models", required=True, nargs="+", help="Model codes to run: cf mf")
    parser.add_argument("--k", required=True, type=int, help="Top-k value for ranking metrics and recommendations")
    parser.add_argument("--output_path", required=True, help="Directory path where outputs will be saved")
    return parser.parse_args()


def _select_best_model(results_df: pd.DataFrame) -> str:
    ranked = results_df.copy()

    ranked["rank_rmse"] = ranked["rmse"].rank(method="min", ascending=True)
    ranked["rank_precision"] = ranked["precision_at_k"].rank(method="min", ascending=False)
    ranked["rank_recall"] = ranked["recall_at_k"].rank(method="min", ascending=False)
    ranked["rank_ndcg"] = ranked["ndcg_at_k"].rank(method="min", ascending=False)

    if "item_coverage" in ranked.columns:
        ranked["rank_item_coverage"] = ranked["item_coverage"].rank(method="min", ascending=False)
    else:
        ranked["rank_item_coverage"] = 0.0

    if "item_gini" in ranked.columns:
        ranked["rank_item_gini"] = ranked["item_gini"].rank(method="min", ascending=True)
    else:
        ranked["rank_item_gini"] = 0.0

    ranked["aggregate_rank"] = (
        ranked["rank_rmse"]
        + ranked["rank_precision"]
        + ranked["rank_recall"]
        + ranked["rank_ndcg"]
        + ranked["rank_item_coverage"]
        + ranked["rank_item_gini"]
    )

    best_row = ranked.sort_values(["aggregate_rank", "rmse", "ndcg_at_k"], ascending=[True, True, False]).iloc[0]
    model_name = str(best_row["model"])

    if model_name == "collaborative_filtering":
        return "cf"
    if model_name == "matrix_factorization":
        return "mf"

    raise ValueError(f"Unsupported model name in evaluation results: {model_name}")


def run_pipeline(data_path: str, model_codes: List[str], k: int, output_path: str) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_all(data_path)

    evaluation_df = run_evaluation(data_dir=data_path, model_codes=model_codes, k=k)
    eval_csv_path = output_dir / "evaluation_results.csv"
    evaluation_df.to_csv(eval_csv_path, index=False)

    plot_paths = generate_all_plots(evaluation_df, output_dir=output_dir, k=k)

    best_model_code = _select_best_model(evaluation_df)
    rec_path = output_dir / "ratings_test_completed.csv"
    generate_recommendations(data_dir=data_path, output_path=rec_path, k=k, model_code=best_model_code)

    print("Pipeline run complete")
    print(f"Evaluation CSV: {eval_csv_path}")
    print(f"Selected best model code: {best_model_code}")
    print(f"Recommendations CSV: {rec_path}")
    print(f"Plots: {plot_paths}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        model_codes=args.models,
        k=args.k,
        output_path=args.output_path,
    )
#CollaborativeFilteringModel = cf
#MatrixFactorizationModel = mf