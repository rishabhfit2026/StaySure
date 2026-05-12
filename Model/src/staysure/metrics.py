import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_target(values: np.ndarray, log_transform: bool) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if log_transform:
        return np.expm1(values)
    return values


def transform_target(values: np.ndarray, log_transform: bool) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if log_transform:
        return np.log1p(values)
    return values


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    denominator = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)
    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "mape": mape,
    }


def confidence_from_neighbors(similarities: np.ndarray, prediction: float, rent_std: float) -> float:
    if similarities.size == 0:
        return 0.55
    similarity_score = float(np.clip(np.mean(similarities[:5]), 0, 1))
    stability = 1.0 / (1.0 + max(rent_std, 0.0) / max(prediction, 1.0))
    return float(np.clip(0.35 + 0.4 * similarity_score + 0.25 * stability, 0.35, 0.95))

