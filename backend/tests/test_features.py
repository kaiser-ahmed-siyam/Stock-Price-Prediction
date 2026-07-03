import pandas as pd

from backend.app.ml.features import build_prediction_row, prepare_training_data


def sample_data():
    rows = []
    for stock in ["AAA", "BBB"]:
        for day in range(1, 9):
            close = 100 + day if stock == "AAA" else 200 + day
            rows.append(
                {
                    "DATE": f"2026-01-{day:02d}",
                    "STOCK": stock,
                    "OPEN_PRICE": close - 1,
                    "HIGH_PRICE": close + 1,
                    "LOW_PRICE": close - 2,
                    "CLOSE_PRICE": close,
                    "VOLUME": 1000 + day,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_training_data_keeps_stock_groups_separate():
    prepared = prepare_training_data(sample_data())
    aaa = prepared.frame[prepared.frame["STOCK"] == "AAA"].iloc[0]
    assert aaa["Close_lag_1"] != 208
    assert set(prepared.features.columns) >= {"STOCK", "CLOSE_PRICE", "Close_lag_1"}


def test_build_prediction_row_uses_history_for_lags():
    row = build_prediction_row(
        sample_data(),
        {
            "stock": "AAA",
            "open_price": 108,
            "high_price": 111,
            "low_price": 107,
            "close_price": 110,
            "volume": 2000,
        },
    )
    assert row.iloc[0]["STOCK"] == "AAA"
    assert row.iloc[0]["Close_lag_1"] == 108
