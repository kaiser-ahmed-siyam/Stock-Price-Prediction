from pydantic import BaseModel, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    stock: str = Field(..., examples=["00DSEX"])
    open_price: float = Field(..., gt=0, examples=[4865.33])
    high_price: float = Field(..., gt=0, examples=[4916.03])
    low_price: float = Field(..., gt=0, examples=[4869.60])
    close_price: float = Field(..., gt=0, examples=[4910.61])
    volume: float = Field(..., ge=0, examples=[3681599000])

    @field_validator("stock")
    @classmethod
    def normalize_stock(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("Stock symbol is required.")
        return value

    @model_validator(mode="after")
    def prices_must_be_consistent(self):
        if self.high_price < self.low_price:
            raise ValueError("High price must be greater than or equal to low price.")
        if not self.low_price <= self.open_price <= self.high_price:
            raise ValueError("Open price must be between low and high price.")
        if not self.low_price <= self.close_price <= self.high_price:
            raise ValueError("Close price must be between low and high price.")
        return self


class PredictionResponse(BaseModel):
    stock: str
    predicted_close_price: float
    input_close_price: float
    expected_change: float
    expected_change_percent: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetadataResponse(BaseModel):
    trained_at: str | None = None
    rows: int | None = None
    stocks: int | None = None
    metrics: dict | None = None
    features: list[str] | None = None
