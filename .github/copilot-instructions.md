# MLOps Assignment 2: Cats vs Dogs Classification

## Project Overview

End-to-end MLOps pipeline demonstrating:
- **M1**: Model Development & Experiment Tracking (MLflow, DVC)
- **M2**: Model Packaging & Containerization (FastAPI, Docker)
- **M3**: CI Pipeline for Build, Test & Image Creation (GitHub Actions)
- **M4**: CD Pipeline & Deployment (Kubernetes, Docker Compose)
- **M5**: Monitoring, Logs & Final Submission (Prometheus, Logging)

## Quick Start

### Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Model Training
```bash
python src/data/preprocessing.py  # Preprocess data
python src/models/train.py        # Train with MLflow tracking
mlflow ui                          # View experiments
```

### Run API Server
```bash
uvicorn src.inference.api:app --reload
# Documentation at http://localhost:8000/docs
```

### Run Tests
```bash
pytest tests/unit/ -v --cov=src
```

### Deploy Locally
```bash
docker-compose -f deploy/docker-compose/docker-compose.yml up -d
python deploy/smoke_tests.py
```

## Project Structure

- `src/data/preprocessing.py` - Data preprocessing & splitting
- `src/models/cnn_model.py` - Model architectures
- `src/models/train.py` - Training script with MLflow
- `src/inference/api.py` - FastAPI inference service
- `tests/unit/` - Unit tests for preprocessing and inference
- `deploy/` - Deployment manifests (K8s, Docker Compose)
- `.github/workflows/ci-cd.yml` - CI/CD pipeline
- `Dockerfile` - Container image definition
- `dvc.yaml` - DVC pipeline configuration

## Key Technologies

- PyTorch / TensorFlow for model development
- MLflow for experiment tracking
- DVC for data versioning
- FastAPI for REST API
- Docker for containerization
- Kubernetes for orchestration
- GitHub Actions for CI/CD
- Pytest for testing
- Prometheus for monitoring

## Data Requirements

Download Cats vs Dogs dataset from Kaggle and place in:
```
data/raw/
├── cats/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── dogs/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Deliverables Checklist

- [x] Git repository with source code
- [x] DVC configuration for data versioning
- [x] MLflow experiment tracking
- [x] FastAPI inference service (health + predict endpoints)
- [x] Pinned requirements.txt
- [x] Dockerfile with layer optimization
- [x] Unit tests with pytest
- [x] GitHub Actions CI pipeline
- [x] Kubernetes deployment manifests
- [x] Docker Compose deployment
- [x] Smoke tests and health checks
- [x] Prometheus monitoring setup
- [x] Complete documentation

## Development Guidelines

### Code Quality
```bash
black src/ tests/          # Format code
flake8 src/ tests/         # Lint
pytest tests/ -v --cov     # Test with coverage
```

### Adding Features
1. Create feature branch
2. Implement and test
3. Update tests
4. Create PR - CI/CD will automatically test

### Model Training
- Pipeline defined in `dvc.yaml`
- Hyperparameters in `params.yaml`
- Run: `dvc repro`

### Deployment
- Local: `docker-compose up`
- K8s: `kubectl apply -f deploy/k8s/deployment.yaml`
- Verify: `python deploy/smoke_tests.py`

## Monitoring & Logs

- API logs are printed to stdout
- Prometheus metrics at `/metrics` endpoint
- MLflow UI: `mlflow ui`
- Docker logs: `docker logs <container-id>`
- K8s logs: `kubectl logs <pod-name>`

## Troubleshooting

**Model not loading?** Check `models/artifacts/model.pt` exists
**CUDA error?** Service auto-falls back to CPU
**API not responding?** Check health: `curl http://localhost:8000/health`
**Pod crash?** Review K8s logs and resource limits

## Performance Targets

- Accuracy: >85%
- Precision: >85%
- Recall: >85%
- F1-Score: >85%
- Inference latency: <500ms
