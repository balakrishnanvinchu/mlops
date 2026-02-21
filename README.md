# MLOps Assignment 2: Cats vs Dogs Binary Image Classification

## Overview

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform. This project demonstrates best practices in model development, experiment tracking, containerization, CI/CD, deployment, and monitoring.

## Project Structure

```
mlops-cats-dogs/
├── data/
│   ├── raw/              # Raw dataset
│   └── processed/        # Preprocessed and split data
├── models/
│   └── artifacts/        # Trained models and metrics
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data preprocessing & splitting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py          # CNN and ResNet models
│   │   └── train.py              # Training script with MLflow
│   └── inference/
│       ├── __init__.py
│       └── api.py                # FastAPI inference service
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   └── test_inference.py
│   └── integration/
├── deploy/
│   ├── k8s/
│   │   └── deployment.yaml       # Kubernetes manifests
│   ├── docker-compose/
│   │   └── docker-compose.yml
│   └── smoke_tests.py            # Post-deployment health checks
├── monitoring/
│   └── prometheus.yml            # Prometheus configuration
├── notebooks/
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD pipeline
├── Dockerfile
├── requirements.txt              # Python dependencies
├── dvc.yaml                      # DVC pipeline config
├── params.yaml                   # Model hyperparameters
├── .gitignore
└── README.md
```

## Modules

### M1: Model Development & Experiment Tracking (10M)

**Objective**: Build baseline model, track experiments, and version all artifacts.

**Tasks completed**:
- Git & DVC setup for code and data versioning
- Data preprocessing script with train/val/test splitting (80/10/10)
- Simple CNN and ResNet50 models for binary classification
- MLflow integration for experiment tracking, metrics, and artifact management

**Key features**:
- DVC pipeline for reproducible data processing and training
- Hyperparameter management via `params.yaml`
- Experiment tracking with MLflow (logs, metrics, confusion matrix, loss curves)
- Model serialization in PyTorch format

### M2: Model Packaging & Containerization (10M)

**Objective**: Package trained model into reproducible containerized service.

**Tasks completed**:
- FastAPI REST inference service with health check and prediction endpoints
- Base64 image encoding for predictions
- `requirements.txt` with pinned versions for reproducibility
- Dockerfile for containerization with layer optimization

**Key endpoints**:
- `GET /health` - Health check
- `GET /info` - Service information
- `POST /predict` - Prediction via base64 image
- `POST /predict-file` - Prediction via file upload
- `GET /metrics` - Request and prediction metrics

### M3: CI Pipeline for Build, Test & Image Creation (10M)

**Objective**: Automated testing, packaging, and container image building.

**Tasks completed**:
- Unit tests for preprocessing and inference modules
- Code quality checks (flake8, black formatting)
- GitHub Actions CI pipeline with:
  - Dependency installation and caching
  - Unit tests with coverage reporting
  - Docker image building and pushing to GitHub Container Registry

### M4: CD Pipeline & Deployment (10M)

**Objective**: Deploy containerized model to target environments.

**Tasks completed**:
- Kubernetes deployment manifests with:
  - Deployment (2 replicas)
  - Service (LoadBalancer)
  - HorizontalPodAutoscaler (2-5 replicas based on CPU/memory)
  - Liveness and readiness probes
- Docker Compose for local deployment
- Smoke tests for post-deployment verification

### M5: Monitoring, Logs & Final Submission (10M)

**Objective**: Monitor deployed model and collect performance data.

**Tasks completed**:
- Request/response logging in FastAPI service
- Metrics collection (request count, prediction count)
- Prometheus integration for metrics scraping
- Post-deployment smoke tests
- Health check endpoints

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for containerization)
- Git & DVC (for versioning)
- kubectl & minikube/kind (for Kubernetes deployment - optional)

### Installation

1. **Clone repository**:
```bash
git clone <repo-url>
cd mlops-cats-dogs
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Initialize DVC** (if not already done):
```bash
dvc init
git add .dvc
```

5. **Download dataset** (Cats and Dogs from Kaggle):
```bash
# Place raw images in data/raw/ with structure:
# data/raw/cats/*.jpg
# data/raw/dogs/*.jpg
```

## Running the Pipeline

### M1: Model Development

1. **Preprocess data**:
```bash
python src/data/preprocessing.py
```

2. **Train model with MLflow tracking**:
```bash
python src/models/train.py
```

View MLflow UI:
```bash
mlflow ui
# Open http://localhost:5000
```

### M2: Local API Testing

1. **Start FastAPI service**:
```bash
uvicorn src.inference.api:app --reload
```

2. **Test endpoints**:
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict-file \
  -F "file=@path/to/image.jpg"
```

API documentation: `http://localhost:8000/docs`

### M3: Running Tests

```bash
# Run all unit tests
pytest tests/unit/ -v --cov=src

# Run specific test
pytest tests/unit/test_preprocessing.py -v
```

### M4: Deployment

**Option 1: Docker Compose**
```bash
docker-compose -f deploy/docker-compose/docker-compose.yml up -d
```

**Option 2: Kubernetes (minikube example)**
```bash
minikube start
docker build -t cats-dogs-classifier:latest .
minikube image load cats-dogs-classifier:latest

kubectl apply -f deploy/k8s/deployment.yaml
kubectl get svc cats-dogs-classifier-service
```

### M5: Smoke Tests

```bash
python deploy/smoke_tests.py --url http://localhost:8000
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) automatically:

1. Runs tests and code quality checks on every push/PR
2. Builds Docker image
3. Pushes to GitHub Container Registry
4. Runs smoke tests on main branch deployments

## Monitoring

View Prometheus metrics:
```bash
# Prometheus UI at http://localhost:9090
# Add data source in Grafana for visualization
```

## Data Versioning with DVC

```bash
# Add dataset to DVC
dvc add data/raw/
git add data/raw/.gitignore
git commit -m "Add raw dataset"

# Track pipeline
dvc repro
```

## Model Artifacts

- **Model weights**: `models/artifacts/model.pt`
- **Metrics**: `models/artifacts/metrics.json`
- **Data stats**: `data/processed/stats.json`
- **MLflow runs**: `mlruns/` directory

## Key Technologies

- **Framework**: PyTorch, TensorFlow
- **API**: FastAPI, Uvicorn
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC, Git
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Testing**: Pytest
- **Monitoring**: Prometheus

## Performance Metrics

Target metrics for test set:
- Accuracy: >85%
- Precision: >85%
- Recall: >85%
- F1-Score: >85%

## Troubleshooting

### Model not found in Docker
Ensure model weights are copied to `models/artifacts/model.pt` before building Docker image.

### CUDA not available
The service will automatically fall back to CPU. Check logs for device confirmation.

### Kubernetes pod crashes
Check resource limits and requests in `deploy/k8s/deployment.yaml`. Adjust if necessary.

## Contributing

1. Create feature branch
2. Make changes and test: `pytest tests/ -v`
3. Push and create pull request
4. CI pipeline automatically tests and builds

## License

Academic Assignment - All rights reserved

## Contact

For questions or issues, please create a GitHub issue in the repository.
