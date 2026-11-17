
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.inference import predict

# Create FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API for predicting customer churn using Machine Learning",
    version="1.0.0"
)


# Input data validation
class CustomerData(BaseModel):

    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Telco Customer Churn API is running"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict_churn(data: CustomerData):

    try:
        # Convert input data into dictionary
        payload = data.dict()

        # Call existing ML inference pipeline
        result = predict(payload)

        return {
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
