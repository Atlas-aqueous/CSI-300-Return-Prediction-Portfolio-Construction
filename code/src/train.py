from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from featurework import (
    SEED,
    build_training_frame,
    data_dir,
    ensure_directories,
    model_dir,
    read_market_data,
    save_json,
    temp_dir,
)


MODEL_PATH = model_dir() / "xgb_return_model.joblib"
REPORT_PATH = temp_dir() / "training_report.json"


def _log(message: str) -> None:
    print(message, flush=True)


def _elapsed(start_time: float) -> str:
    return f"{time.time() - start_time:.2f}s"


def _fill_features(frame, feature_columns, fill_values):
    filled = frame.copy()
    for column in feature_columns:
        filled[column] = filled[column].fillna(fill_values[column])
    return filled


def _fit_mlp(train_x, train_y, label: str) -> tuple[StandardScaler, MLPRegressor]:
    _log(f"[INFO] Scaling features for {label} MLPRegressor...")
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(train_x)
    _log(f"[INFO] Training {label} MLPRegressor...")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
        verbose=True,
    )
    model.fit(scaled_x, train_y)
    _log(f"[INFO] {label} MLPRegressor finished. n_iter={getattr(model, 'n_iter_', 'NA')}")
    return scaler, model


def _daily_topk_return(frame, score_column: str, top_k: int = 5) -> float:
    daily_returns = (
        frame.groupby("trade_date", group_keys=False)
        .apply(lambda day_frame: float(day_frame.nlargest(top_k, score_column)["target"].mean()))
        .to_numpy(dtype=float)
    )
    return float(np.nanmean(daily_returns))


def _rmse(y_true, y_pred) -> float:
    try:
        from sklearn.metrics import root_mean_squared_error

        return float(root_mean_squared_error(y_true, y_pred))
    except ImportError:
        return float(mean_squared_error(y_true, y_pred, squared=False))


def _print_frame_summary(name: str, frame, feature_columns=None) -> None:
    _log(f"[INFO] {name} rows: {len(frame)}")
    if "trade_date" in frame.columns and len(frame) > 0:
        trade_dates = frame["trade_date"]
        _log(f"[INFO] {name} date range: {trade_dates.min()} -> {trade_dates.max()}")
        _log(f"[INFO] {name} trading days: {trade_dates.nunique()}")
    if "stock_id" in frame.columns and len(frame) > 0:
        _log(f"[INFO] {name} stocks: {frame['stock_id'].astype(str).nunique()}")
    if feature_columns is not None:
        _log(f"[INFO] {name} feature count: {len(feature_columns)}")


def train_and_save_model() -> dict:
    total_start = time.time()
    _log("============================================================")
    _log("[version_zero] Training started")
    _log("============================================================")
    _log(f"[INFO] Random seed: {SEED}")
    _log(f"[INFO] Train data path: {data_dir() / 'train.csv'}")
    _log(f"[INFO] Model output path: {MODEL_PATH}")
    _log(f"[INFO] Report output path: {REPORT_PATH}")

    ensure_directories()

    stage_start = time.time()
    _log("------------------------------------------------------------")
    _log("[INFO] Loading market data...")
    train_df = read_market_data(data_dir() / "train.csv")
    _log(f"[INFO] Raw train.csv shape: {train_df.shape}")
    if "trade_date" in train_df.columns:
        _log(f"[INFO] Raw date range: {train_df['trade_date'].min()} -> {train_df['trade_date'].max()}")
    if "stock_id" in train_df.columns:
        _log(f"[INFO] Raw stock count: {train_df['stock_id'].astype(str).nunique()}")
    _log(f"[INFO] Loading market data finished in {_elapsed(stage_start)}")

    stage_start = time.time()
    _log("------------------------------------------------------------")
    _log("[INFO] Building training frame and features...")
    training_frame, feature_columns = build_training_frame(train_df)
    _print_frame_summary("training_frame", training_frame, feature_columns)
    _log(f"[INFO] Building features finished in {_elapsed(stage_start)}")

    unique_dates = sorted(training_frame["trade_date"].unique())
    validation_days = min(80, max(20, len(unique_dates) // 6))
    validation_dates = set(unique_dates[-validation_days:])
    _log("------------------------------------------------------------")
    _log("[INFO] Splitting internal train/validation set...")
    _log(f"[INFO] Total trading days: {len(unique_dates)}")
    _log(f"[INFO] Validation days: {validation_days}")
    _log(f"[INFO] Validation date range: {min(validation_dates)} -> {max(validation_dates)}")

    train_part = training_frame[~training_frame["trade_date"].isin(validation_dates)].copy()
    valid_part = training_frame[training_frame["trade_date"].isin(validation_dates)].copy()
    _print_frame_summary("train_part", train_part, feature_columns)
    _print_frame_summary("valid_part", valid_part, feature_columns)

    stage_start = time.time()
    _log("------------------------------------------------------------")
    _log("[INFO] Filling missing feature values with train median...")
    fill_values = train_part[feature_columns].median(numeric_only=True).to_dict()
    train_part = _fill_features(train_part, feature_columns, fill_values)
    valid_part = _fill_features(valid_part, feature_columns, fill_values)
    _log(f"[INFO] Missing value filling finished in {_elapsed(stage_start)}")

    _log("------------------------------------------------------------")
    _log("[INFO] Training validation XGBRegressor...")
    stage_start = time.time()
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(
        train_part[feature_columns],
        train_part["target"],
        sample_weight=train_part["sample_weight"],
        eval_set=[(valid_part[feature_columns], valid_part["target"])],
        verbose=50,
    )
    _log(f"[INFO] Validation XGBRegressor finished in {_elapsed(stage_start)}")

    stage_start = time.time()
    mlp_scaler, mlp_model = _fit_mlp(train_part[feature_columns], train_part["target"], "validation")
    _log(f"[INFO] Validation MLPRegressor finished in {_elapsed(stage_start)}")

    stage_start = time.time()
    _log("------------------------------------------------------------")
    _log("[INFO] Evaluating validation predictions...")
    valid_part["xgb_prediction"] = model.predict(valid_part[feature_columns])
    valid_part["mlp_prediction"] = mlp_model.predict(mlp_scaler.transform(valid_part[feature_columns]))
    valid_part["prediction"] = 0.65 * valid_part["xgb_prediction"] + 0.35 * valid_part["mlp_prediction"]
    validation_report = {
        "train_rows": int(len(train_part)),
        "validation_rows": int(len(valid_part)),
        "feature_count": int(len(feature_columns)),
        "validation_rmse": _rmse(valid_part["target"], valid_part["prediction"]),
        "validation_xgb_rmse": _rmse(valid_part["target"], valid_part["xgb_prediction"]),
        "validation_mlp_rmse": _rmse(valid_part["target"], valid_part["mlp_prediction"]),
        "validation_pred_top5_return": _daily_topk_return(valid_part, "prediction", top_k=5),
        "validation_true_top5_return": _daily_topk_return(valid_part, "target", top_k=5),
        "validation_random_return": float(valid_part.groupby("trade_date")["target"].mean().mean()),
        "training_start": str(training_frame["trade_date"].min().date()),
        "training_end": str(training_frame["trade_date"].max().date()),
    }
    _log(f"[METRIC] validation_rmse: {validation_report['validation_rmse']:.6f}")
    _log(f"[METRIC] validation_xgb_rmse: {validation_report['validation_xgb_rmse']:.6f}")
    _log(f"[METRIC] validation_mlp_rmse: {validation_report['validation_mlp_rmse']:.6f}")
    _log(f"[METRIC] validation_pred_top5_return: {validation_report['validation_pred_top5_return']:.6f}")
    _log(f"[METRIC] validation_true_top5_return: {validation_report['validation_true_top5_return']:.6f}")
    _log(f"[METRIC] validation_random_return: {validation_report['validation_random_return']:.6f}")
    _log(f"[INFO] Validation evaluation finished in {_elapsed(stage_start)}")

    stage_start = time.time()
    _log("------------------------------------------------------------")
    _log("[INFO] Rebuilding full training frame for final models...")
    full_fill_values = training_frame[feature_columns].median(numeric_only=True).to_dict()
    full_frame = _fill_features(training_frame, feature_columns, full_fill_values)
    _print_frame_summary("full_frame", full_frame, feature_columns)
    _log(f"[INFO] Full frame preparation finished in {_elapsed(stage_start)}")

    _log("------------------------------------------------------------")
    _log("[INFO] Training final XGBRegressor on full training frame...")
    stage_start = time.time()
    final_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=4,
    )
    final_model.fit(
        full_frame[feature_columns],
        full_frame["target"],
        sample_weight=full_frame["sample_weight"],
        verbose=False,
    )
    _log(f"[INFO] Final XGBRegressor finished in {_elapsed(stage_start)}")

    stage_start = time.time()
    _log("------------------------------------------------------------")
    _log("[INFO] Training final MLPRegressor on full training frame...")
    full_mlp_scaler, full_mlp_model = _fit_mlp(full_frame[feature_columns], full_frame["target"], "final")
    _log(f"[INFO] Final MLPRegressor finished in {_elapsed(stage_start)}")

    stage_start = time.time()
    _log("------------------------------------------------------------")
    _log("[INFO] Saving model bundle and training report...")
    bundle = {
        "xgb_model": final_model,
        "mlp_model": full_mlp_model,
        "mlp_scaler": full_mlp_scaler,
        "feature_columns": feature_columns,
        "fill_values": full_fill_values,
        "training_report": validation_report,
    }
    joblib.dump(bundle, MODEL_PATH)
    save_json(validation_report, REPORT_PATH)
    _log(f"[INFO] Model saved to: {MODEL_PATH}")
    _log(f"[INFO] Report saved to: {REPORT_PATH}")
    _log(f"[INFO] Saving finished in {_elapsed(stage_start)}")

    _log("------------------------------------------------------------")
    _log(f"[version_zero] Training finished in {_elapsed(total_start)}")
    _log("============================================================")
    return validation_report


def main() -> None:
    report = train_and_save_model()
    print(f"模型已保存到: {MODEL_PATH}", flush=True)
    print(f"验证集 RMSE: {report['validation_rmse']:.6f}", flush=True)
    print(f"验证集 Top5 平均收益: {report['validation_pred_top5_return']:.6f}", flush=True)
    print(f"验证集随机组合收益: {report['validation_random_return']:.6f}", flush=True)


if __name__ == "__main__":
    main()
