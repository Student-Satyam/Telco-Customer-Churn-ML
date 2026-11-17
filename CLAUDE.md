# CLAUDE.md

This file provides project-specific guidance for Claude Code and other AI coding assistants working with this repository.

# Project Overview

Telco Customer Churn Prediction is a machine learning application that predicts whether a telecommunications customer is likely to discontinue their service.

The project uses:
- XGBoost for customer churn classification.
- Pandas and NumPy for data processing.
- MLflow for experiment tracking and model artifacts.
- FastAPI for backend inference.
- Streamlit for the interactive frontend.
- Docker for containerization.
- GitHub Actions for automated project checks.

The application uses separate frontend and backend services.

**Important architecture rule:** Streamlit communicates with FastAPI through HTTP requests. Keep model inference centralized in `src/serving/inference.py`.

# Technology Stack

- **Language:** Python 3.11
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Experiment Tracking:** MLflow
- **API Server:** Uvicorn
- **Containerization:** Docker
- **Testing:** Pytest
- **CI/CD:** GitHub Actions

# Project Architecture

```text
Customer
   |
   v
Streamlit Frontend
   |
   | HTTP POST request
   v
FastAPI Backend
   |
   v
Input Validation
   |
   v
Inference Module
   |
   v
Trained XGBoost Model
   |
   v
Prediction Response
   |
   v
Streamlit Frontend
```

## Main Application Files

| File | Responsibility |
|---|---|
| `frontend/streamlit_app.py` | User interface |
| `src/app/main.py` | FastAPI application and API endpoints |
| `src/serving/inference.py` | Model loading and prediction logic |
| `src/serving/model/deployment_model/` | Deployment-ready model artifacts |
| `src/utils/validate_data.py` | Data validation utilities |
| `scripts/run_pipeline.py` | Machine learning pipeline entry point |
| `.github/workflows/ci.yml` | GitHub Actions workflow |

# Commands

## Local Development Setup

Create a virtual environment:

```powershell
python -m venv venv
```

Activate on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run FastAPI

Start the backend:

```powershell
uvicorn src.app.main:app --reload
```

Backend address:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Run Streamlit

Open a second terminal, activate the virtual environment, and run:

```powershell
streamlit run frontend/streamlit_app.py
```

Frontend address:

```text
http://localhost:8501
```

The frontend uses the `API_URL` environment variable to communicate with the backend.

Default local prediction endpoint:

```text
http://127.0.0.1:8000/predict
```

Do not hardcode a cloud URL into the frontend. Configure the URL through the environment variable when deploying.

# API Endpoints

The FastAPI application provides:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application root |
| GET | `/health` | Health check |
| POST | `/predict` | Customer churn prediction |
| GET | `/docs` | Swagger documentation |

Always inspect `src/app/main.py` and the Pydantic request model before changing API request or response structures.

Do not assume the frontend and backend share the same process.

# Machine Learning Pipeline

The training pipeline entry point is:

```text
scripts/run_pipeline.py
```

The general workflow includes:

1. Load the customer churn dataset.
2. Validate the data.
3. Perform data preprocessing.
4. Apply feature engineering.
5. Train the XGBoost classification model.
6. Evaluate model performance.
7. Track experiment information and artifacts using MLflow.

Before running or modifying pipeline commands, inspect the current script and its argument definitions.

Dataset location:

```text
src/data/raw/Telco-Customer-Churn.csv
```

# Feature Engineering

Training and inference must use consistent feature transformations.

The original feature-engineering implementation is located at:

```text
src/features/build_features.py
```

Inspect the current file before modifying it.

Important requirements:

- Maintain consistent binary feature mappings.
- Apply categorical transformations consistently.
- Preserve the expected feature columns.
- Preserve the exact feature order used during training.
- Avoid changing preprocessing logic without checking its impact on predictions.

# Model Loading and Serving

The centralized inference module is:

```text
src/serving/inference.py
```

The deployment-ready model artifacts are stored in:

```text
src/serving/model/deployment_model/
```

Important deployment artifacts include:

- `model.pkl`
- `MLmodel`
- `feature_columns.txt`
- `conda.yaml`
- `python_env.yaml`
- `requirements.txt`
- `metadata/`

The Docker API image copies the deployment artifacts to:

```text
/app/model/
```

The inference module supports selecting the model directory based on the available environment:

1. Docker model directory: `/app/model`
2. Repository deployment model directory: `src/serving/model/deployment_model`
3. Original local MLflow model artifact directory as fallback

Do not delete or overwrite existing historical MLflow model runs without explicit approval.

The inference module uses the saved feature-column list to align incoming customer data with the model's expected input structure.

**Important:** The current inference implementation uses a classification threshold of `0.50`. Do not change this threshold unless the model configuration and prediction behavior have been reviewed and tested.

Load the model once during application startup where supported. Avoid repeatedly loading model artifacts for every prediction request.

# MLflow Integration

MLflow is used for experiment tracking and model artifact management.

Original tracking location:

```text
mlruns/
```

MLflow may contain:

- Trained model artifacts.
- Model configuration.
- Feature-column information.
- Experiment metrics.
- Run parameters.
- Training metadata.

Tracked metrics in the original pipeline include:

- Precision
- Recall
- F1-score
- ROC-AUC
- Training time
- Prediction time
- Data quality status

Inspect the current training configuration before changing experiment names, metric definitions, or model parameters.

The deployment model in `src/serving/model/deployment_model/` must remain available independently of the local `mlruns/` directory.

# Data Validation

Data validation utilities are located in:

```text
src/utils/validate_data.py
```

Inspect the implementation before changing validation rules.

The original validation logic includes checks such as:

- Customer ID presence.
- Valid gender values.
- Numeric ranges for tenure and charges.
- Dataset quality requirements.

Keep validation consistent with the training and inference workflows.

# Docker Configuration

The project has two separate Dockerfiles.

## FastAPI Backend

Dockerfile:

```text
Dockerfile.api
```

Build command:

```powershell
docker build -f Dockerfile.api -t telco-churn-api:latest .
```

The API container:
- Uses Python 3.11 slim.
- Runs the FastAPI application through Uvicorn.
- Exposes port 8000 for local development.
- Copies deployment model artifacts into `/app/model/`.

## Streamlit Frontend

Dockerfile:

```text
dockerfile
```

Build command:

```powershell
docker build -f dockerfile -t telco-churn-streamlit:latest .
```

The Streamlit container:
- Runs the Streamlit application.
- Uses port 8501 for local development.
- Communicates with FastAPI using `API_URL`.

## Docker Network

The two containers communicate through:

```text
telco-network
```

Create the network if it does not already exist:

```powershell
docker network create telco-network
```

Start the API container:

```powershell
docker run -d --name telco-churn-api --network telco-network -p 8000:8000 telco-churn-api:latest
```

Start the Streamlit container:

```powershell
docker run -d --name telco-churn-streamlit --network telco-network -e API_URL=http://telco-churn-api:8000/predict -p 8501:8501 telco-churn-streamlit:latest
```

If containers with these names already exist, remove or rename the old containers before creating replacements.

Verify containers:

```powershell
docker ps
```

View logs:

```powershell
docker logs telco-churn-api
docker logs telco-churn-streamlit
```

Local application URLs:

- Streamlit: `http://localhost