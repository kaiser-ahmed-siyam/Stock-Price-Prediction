from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "DSE Stock Price Prediction"
    app_version: str = "0.1.0"
    data_path: Path = Field(default=ROOT_DIR / "data" / "dse_clean_master_dataset_2026.csv")
    artifact_dir: Path = Field(default=ROOT_DIR / "backend" / "artifacts")
    model_file: str = "model.joblib"
    metadata_file: str = "metadata.json"
    cors_origins: list[str] = ["*"]

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / self.model_file

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / self.metadata_file


@lru_cache
def get_settings() -> Settings:
    return Settings()
