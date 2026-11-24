from __future__ import annotations

import joblib
import pandas as pd

from featurework import (
    build_portfolio,
    data_dir,
    ensure_directories,
    model_dir,
    output_dir,
    prepare_inference_frame,
    project_root,
    read_market_data,
    save_json,
    save_result,
    temp_dir,
)
from train import train_and_save_model


MODEL_PATH = model_dir() / "xgb_return_model.joblib"
RESULT_PATH = output_dir() / "result.csv"
DETAIL_PATH = temp_dir() / "prediction_details.csv"
META_PATH = temp_dir() / "prediction_meta.json"


def _predict_with_model(train_df: pd.DataFrame, history_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MODEL_PATH.exists():
        train_and_save_model()

    bundle = joblib.load(MODEL_PATH)
    candidates = prepare_inference_frame(train_df, history_df, bundle["feature_columns"])
    for column in bundle["feature_columns"]:
        candidates[column] = candidates[column].fillna(bundle["fill_values"][column])

    candidates["xgb_score"] = bundle["xgb_model"].predict(candidates[bundle["feature_columns"]])
    candidates["mlp_score"] = bundle["mlp_model"].predict(bundle["mlp_scaler"].transform(candidates[bundle["feature_columns"]]))
    candidates["quality"] = (
        0.35 * candidates["pct_vs_market"].fillna(0.0)
        + 0.25 * candidates["ret_5"].fillna(0.0)
        + 0.20 * candidates["ret_20"].fillna(0.0)
        + 0.20 * candidates["intraday_strength"].fillna(0.0)
    )
    candidates["volatility"] = candidates["pct_std_20"].fillna(candidates["pct_std_10"]).fillna(0.08).abs()
    candidates["score"] = (
        0.55 * candidates["xgb_score"]
        + 0.25 * candidates["mlp_score"]
        + 0.20 * candidates["quality"]
        - 0.15 * candidates["volatility"]
    )
    portfolio = build_portfolio(candidates[["stock_id", "score", "volatility", "quality"]], mode="model", top_k=5)
    details = candidates[["stock_id", "trade_date", "score", "xgb_score", "mlp_score", "quality", "volatility", "pct_vs_market", "ret_5", "ret_20"]].sort_values(
        "score",
        ascending=False,
    )
    return portfolio, details


def main() -> None:
    ensure_directories()

    # Official submission mode:
    # Only use historical data from train.csv. Do not read or use test.csv,
    # so the generated result.csv is determined by the latest date in train.csv.
    train_df = read_market_data(data_dir() / "train.csv")
    history_df = train_df.copy()
    history_mode = "train_only_official"

    portfolio, details = _predict_with_model(train_df, history_df)
    mode = f"model_prediction_{history_mode}"

    save_result(portfolio, RESULT_PATH)
    details.to_csv(DETAIL_PATH, index=False, encoding="utf-8")
    save_json(
        {
            "mode": mode,
            "result_path": RESULT_PATH.relative_to(project_root()).as_posix(),
            "detail_path": DETAIL_PATH.relative_to(project_root()).as_posix(),
            "selected_count": int(len(portfolio)),
            "weight_sum": float(portfolio["weight"].sum()) if not portfolio.empty else 0.0,
        },
        META_PATH,
    )
    print(f"推理模式: {mode}")
    print(f"结果已保存到: {RESULT_PATH}")
    print(portfolio if not portfolio.empty else "当前最优策略为空仓，结果文件仅保留表头。")


if __name__ == "__main__":
    main()
