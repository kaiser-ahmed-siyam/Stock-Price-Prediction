from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import ROOT_DIR, get_settings
from backend.app.services.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    service = PredictionService(settings)
    app.state.settings = settings
    app.state.prediction_service = service
    service.load()
    yield


app = FastAPI(
    title="DSE Stock Price Prediction API",
    version=get_settings().app_version,
    description="FastAPI service for predicting the next closing price from DSE market inputs.",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api", tags=["prediction"])
app.mount("/", StaticFiles(directory=ROOT_DIR / "frontend" / "static", html=True), name="frontend")
