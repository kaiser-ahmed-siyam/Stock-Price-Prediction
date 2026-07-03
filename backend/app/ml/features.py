from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


LAG_DAYS = (1, 2, 3, 4, 5)

NUMERIC_FEATURES = [
    "OPEN_PRICE",
    "HIGH_PRICE",
    "LOW_PRICE",
    "CLOSE_PRICE",
    "VOLUME",
    "RETURN",
    "VOLATILITY_7D",
    "PRICE_CHANGE",
    "MA_7",
    "MA_30",
    "MOMENTUM",
    *[f"Close_lag_{lag}" for lag in LAG_DAYS],
]

CATEGORICAL_FEATURES = ["STOCK"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = ["DATE", "STOCK", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "VOLUME"]


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [column.strip().upper() for column in cleaned.columns]
    return cleaned


def validate_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_stock_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    validate_columns(df)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE", "STOCK", "CLOSE_PRICE"]).copy()
    numeric_columns = [column for column in df.columns if column not in {"DATE", "STOCK"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["STOCK", "DATE"]).reset_index(drop=True)


def add_time_series_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    data = normalize_columns(df)
    validate_columns(data)
    data["DATE"] = pd.to_datetime(data["DATE"], errors="coerce")
    data = data.dropna(subset=["DATE", "STOCK", "CLOSE_PRICE"]).copy()
    data = data.sort_values(["STOCK", "DATE"]).reset_index(drop=True)

    grouped = data.groupby("STOCK", sort=False)
    data["PRICE_CHANGE"] = data["CLOSE_PRICE"] - data["OPEN_PRICE"]
    data["RETURN"] = grouped["CLOSE_PRICE"].pct_change()
    data["VOLATILITY_7D"] = grouped["RETURN"].transform(lambda values: values.rolling(7, min_periods=2).std())
    data["MA_7"] = grouped["CLOSE_PRICE"].transform(lambda values: values.rolling(7, min_periods=1).mean())
    data["MA_30"] = grouped["CLOSE_PRICE"].transform(lambda values: values.rolling(30, min_periods=1).mean())
    data["MOMENTUM"] = data["CLOSE_PRICE"] - grouped["CLOSE_PRICE"].shift(5)

    for lag in LAG_DAYS:
        data[f"Close_lag_{lag}"] = grouped["CLOSE_PRICE"].shift(lag)

    if include_target:
        data["Target"] = grouped["CLOSE_PRICE"].shift(-1)
        data = data.dropna(subset=["Target"]).copy()

    data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return data


def prepare_training_data(df: pd.DataFrame) -> PreparedData:
    featured = add_time_series_features(df, include_target=True)
    featured[NUMERIC_FEATURES] = featured.groupby("STOCK", sort=False)[NUMERIC_FEATURES].ffill()
    featured[NUMERIC_FEATURES] = featured[NUMERIC_FEATURES].fillna(featured[NUMERIC_FEATURES].median(numeric_only=True))
    featured = featured.dropna(subset=FEATURES + ["Target"]).copy()
    featured = featured.sort_values(["DATE", "STOCK"]).reset_index(drop=True)
    return PreparedData(frame=featured, features=featured[FEATURES], target=featured["Target"])


def build_prediction_row(history: pd.DataFrame, user_row: dict) -> pd.DataFrame:
    stock = str(user_row["stock"]).strip().upper()
    incoming = {
        "DATE": pd.Timestamp.utcnow().tz_localize(None),
        "STOCK": stock,
        "OPEN_PRICE": float(user_row["open_price"]),
        "HIGH_PRICE": float(user_row["high_price"]),
        "LOW_PRICE": float(user_row["low_price"]),
        "CLOSE_PRICE": float(user_row["close_price"]),
        "VOLUME": float(user_row["volume"]),
    }

    history = normalize_columns(history)
    stock_history = history[history["STOCK"].astype(str).str.upper() == stock].copy()
    if stock_history.empty:
        raise ValueError(f"Stock '{stock}' is not available in the training history.")

    combined = pd.concat([stock_history[REQUIRED_COLUMNS], pd.DataFrame([incoming])], ignore_index=True)
    featured = add_time_series_features(combined, include_target=False)
    row = featured.iloc[[-1]][FEATURES].copy()
    row[NUMERIC_FEATURES] = row[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return row
