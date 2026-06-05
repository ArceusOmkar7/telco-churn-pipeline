from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.schemas import CustomerFeatures, PredictionResponse
from api.predict import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model and preprocessor
    model_service.load_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    result = model_service.predict(features)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
