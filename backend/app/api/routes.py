from fastapi import APIRouter, HTTPException, Request

from backend.app.models.schemas import HealthResponse, MetadataResponse, PredictionRequest, PredictionResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = request.app.state.prediction_service
    settings = request.app.state.settings
    return HealthResponse(status="ok", model_loaded=service.is_loaded, version=settings.app_version)


@router.get("/metadata", response_model=MetadataResponse)
def metadata(request: Request) -> MetadataResponse:
    service = request.app.state.prediction_service
    return MetadataResponse(**service.metadata)


@router.get("/stocks", response_model=list[str])
def stocks(request: Request) -> list[str]:
    service = request.app.state.prediction_service
    return service.stock_symbols()


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    service = request.app.state.prediction_service
    try:
        return service.predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
