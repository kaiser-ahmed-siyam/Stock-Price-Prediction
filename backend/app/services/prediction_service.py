from __future__ import annotations

import json
from pathlib import Path

import joblib

from backend.app.core.config import Settings
from backend.app.ml.features import build_prediction_row, load_stock_data
from backend.app.models.schemas import PredictionRequest, PredictionResponse


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.history = None
        self.metadata: dict = {}

    def load(self) -> None:
        if not self.settings.model_path.exists():
            from backend.app.training.train import train

            train(self.settings.data_path, self.settings.artifact_dir)

        self.model = joblib.load(self.settings.model_path)
        self.history = load_stock_data(str(self.settings.data_path))
        if self.settings.metadata_path.exists():
            self.metadata = json.loads(self.settings.metadata_path.read_text(encoding="utf-8"))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.history is not None

    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        if not self.is_loaded:
            self.load()

        row = build_prediction_row(self.history, payload.model_dump())
        predicted_price = float(self.model.predict(row)[0])
        change = predicted_price - payload.close_price
        change_percent = (change / payload.close_price) * 100
        return PredictionResponse(
            stock=payload.stock,
            predicted_close_price=round(predicted_price, 4),
            input_close_price=round(payload.close_price, 4),
            expected_change=round(change, 4),
            expected_change_percent=round(change_percent, 4),
            model_version=self.metadata.get("trained_at", "unversioned"),
        )

    def stock_symbols(self) -> list[str]:
        if self.history is None:
            self.history = load_stock_data(str(self.settings.data_path))
        return sorted(self.history["STOCK"].astype(str).str.upper().unique().tolist())


def artifact_exists(settings: Settings) -> bool:
    return Path(settings.model_path).exists()
