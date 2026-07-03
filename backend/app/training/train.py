from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from backend.app.core.config import get_settings
from backend.app.ml.features import FEATURES, load_stock_data, prepare_training_data
from backend.app.ml.model import build_model, regression_metrics


def train(data_path: Path | None = None, artifact_dir: Path | None = None) -> dict:
    settings = get_settings()
    data_path = data_path or settings.data_path
    artifact_dir = artifact_dir or settings.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw = load_stock_data(str(data_path))
    prepared = prepare_training_data(raw)

    split_idx = int(len(prepared.features) * 0.8)
    x_train = prepared.features.iloc[:split_idx]
    x_test = prepared.features.iloc[split_idx:]
    y_train = prepared.target.iloc[:split_idx]
    y_test = prepared.target.iloc[split_idx:]

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = regression_metrics(y_test, predictions)

    joblib.dump(model, artifact_dir / settings.model_file)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "rows": int(len(prepared.frame)),
        "stocks": int(prepared.frame["STOCK"].nunique()),
        "features": FEATURES,
        "metrics": metrics,
    }
    (artifact_dir / settings.metadata_file).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
