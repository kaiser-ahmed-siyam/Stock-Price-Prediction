# DSE Stock Price Prediction

FastAPI web service for predicting the next DSE closing price from user-submitted market values.

## Project Structure

- `backend/app/api` - API routes.
- `backend/app/ml` - feature engineering and model pipeline.
- `backend/app/services` - prediction service and artifact loading.
- `backend/app/training` - reproducible model training entry point.
- `frontend/static` - static frontend served by FastAPI.
- `research` - notebooks and experiments.
- `data` - source CSV data.

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m backend.app.training.train
uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000` for the frontend and `http://127.0.0.1:8000/docs` for the API docs.

## API

`POST /api/predict`

```json
{
  "stock": "00DSEX",
  "open_price": 4865.33,
  "high_price": 4916.03,
  "low_price": 4869.60,
  "close_price": 4910.61,
  "volume": 3681599000
}
```

The backend computes rolling features and lag values from the stored stock history, then returns the predicted next close price.

## Website
[Stock Price Prediction](https://stock-price-prediction-iw1s.onrender.com/) Render Deployment

This repository includes `render.yaml`. Push the project to GitHub, create a new Blueprint on Render, and Render will run:

```bash
pip install -r requirements.txt && python -m backend.app.training.train
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Render now defaults new Python services to Python 3.14.x. This project pins Python 3.11.9 in `.python-version` and `render.yaml` so scientific packages such as NumPy, SciPy, pandas, and scikit-learn install from stable wheels instead of compiling from source.


## Notebook Improvements Applied

- Target and lag features are computed per stock instead of across the full dataset.
- The scaler/preprocessor is fitted only on training data through a scikit-learn pipeline.
- The deployed model artifact is the fitted pipeline, not an unfitted estimator.
- API input validation is handled with Pydantic.
- Frontend, backend, training, experiments, versioning, and requirements are separated.
