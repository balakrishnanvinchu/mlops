# MLOps Assignment 2 - Project Setup Complete ✓

## 🎉 Summary

Your complete end-to-end MLOps pipeline has been created with all 5 modules fully implemented!

**Location**: `C:\Users\balak\mlops-cats-dogs`

## What's Included

### ✅ M1: Model Development & Experiment Tracking (10 marks)
- Data preprocessing with train/val/test splitting (80/10/10)
- Simple CNN and ResNet50 models
- MLflow integration for experiment tracking
- DVC pipeline configuration
- Parameter management via `params.yaml`

### ✅ M2: Model Packaging & Containerization (10 marks)
- FastAPI REST inference service
- 4 endpoints: `/health`, `/info`, `/predict`, `/predict-file`
- Dockerfile with optimization
- Pinned dependencies in `requirements.txt`

### ✅ M3: CI Pipeline for Build, Test & Image Creation (10 marks)
- Unit tests for preprocessing and inference
- Code quality checks (black, flake8)
- GitHub Actions CI pipeline
- Automated Docker image build and push

### ✅ M4: CD Pipeline & Deployment (10 marks)
- Kubernetes deployment manifests (Deployment, Service, HPA)
- Docker Compose configuration
- Health checks and readiness probes
- Auto-scaling configuration

### ✅ M5: Monitoring, Logs & Final Submission (10 marks)
- Request/response logging in API
- Prometheus metrics integration
- Post-deployment smoke tests
- Health monitoring endpoints

## 📁 Project Structure

```
mlops-cats-dogs/
├── src/                          # Source code
│   ├── data/preprocessing.py    # Data preprocessing
│   ├── models/
│   │   ├── cnn_model.py         # Model architectures
│   │   └── train.py             # Training with MLflow
│   └── inference/api.py         # FastAPI service
├── tests/                       # Test suite
│   └── unit/
│       ├── test_preprocessing.py
│       └── test_inference.py
├── deploy/                      # Deployment configs
│   ├── k8s/deployment.yaml     # Kubernetes manifests
│   ├── docker-compose/          # Docker Compose
│   └── smoke_tests.py           # Health checks
├── monitoring/prometheus.yml    # Prometheus config
├── .github/workflows/ci-cd.yml # GitHub Actions pipeline
├── Dockerfile                   # Container definition
├── requirements.txt             # Dependencies (pinned)
├── params.yaml                  # Model hyperparameters
├── dvc.yaml                     # DVC pipeline
├── Makefile                     # Convenience commands
└── README.md                    # Full documentation
```

## 🚀 Next Steps (Quick Start)

### 1. Download Dataset
```bash
# Download Cats vs Dogs dataset from Kaggle
# Place in this structure:
# data/raw/cats/*.jpg
# data/raw/dogs/*.jpg
```

### 2. Setup Python Environment
```bash
cd C:\Users\balak\mlops-cats-dogs
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Preprocess Data
```bash
python src/data/preprocessing.py
```

### 4. Train Model
```bash
python src/models/train.py
# View experiments: mlflow ui
```

### 5. Test Inference Service
```bash
uvicorn src.inference.api:app --reload
# API Docs: http://localhost:8000/docs
```

### 6. Run Tests
```bash
pytest tests/unit/ -v --cov=src
```

### 7. Deploy Locally
```bash
docker-compose -f deploy/docker-compose/docker-compose.yml up -d
python deploy/smoke_tests.py
```

## 📚 Key Files for Each Module

| Module | Key Files |
|--------|-----------|
| M1 | `src/data/preprocessing.py`, `src/models/train.py`, `dvc.yaml`, `params.yaml` |
| M2 | `src/inference/api.py`, `requirements.txt`, `Dockerfile` |
| M3 | `tests/unit/*.py`, `.github/workflows/ci-cd.yml` |
| M4 | `deploy/k8s/deployment.yaml`, `deploy/docker-compose/docker-compose.yml` |
| M5 | `deploy/smoke_tests.py`, `monitoring/prometheus.yml` |

## 🛠️ Useful Commands (Makefile)

```bash
make install        # Install dependencies
make test           # Run tests with coverage
make preprocess     # Run data preprocessing
make train          # Train model
make run-api        # Start FastAPI server
make docker-build   # Build Docker image
make deploy-compose # Deploy with Docker Compose
make smoke-tests    # Run post-deployment tests
make lint           # Code quality checks
make clean          # Clean artifacts
```

## 📋 Deliverables Checklist

For final submission, include:

- [x] GitHub repository with all source code ✓
- [x] Git history with initial commit ✓
- [x] DVC configuration for data versioning ✓
- [x] MLflow experiment tracking setup ✓
- [x] FastAPI inference service with endpoints ✓
- [x] Comprehensive unit tests ✓
- [x] GitHub Actions CI/CD pipeline ✓
- [x] Docker containerization ✓
- [x] Kubernetes deployment manifests ✓
- [x] Docker Compose deployment ✓
- [x] Monitoring and health checks ✓
- [x] Pinned dependencies ✓
- [x] Complete documentation ✓

## 🎬 For Screen Recording Demo

1. Show Git history: `git log --oneline`
2. Run preprocessing: `python src/data/preprocessing.py`
3. Train model: `python src/models/train.py`
4. Show MLflow UI: `mlflow ui`
5. Start API: `uvicorn src.inference.api:app --reload`
6. Test endpoint: `curl http://localhost:8000/health`
7. Build Docker: `docker build -t cats-dogs-classifier:latest .`
8. Run Docker Compose: `docker-compose -f deploy/docker-compose/docker-compose.yml up -d`
9. Run smoke tests: `python deploy/smoke_tests.py`

## 📝 Important Notes

1. **Dataset**: Download from Kaggle and place in `data/raw/` before preprocessing
2. **Model Path**: After training, model is saved at `models/artifacts/model.pt`
3. **MLflow**: Run `mlflow ui` to see experiment tracking (port 5000)
4. **Docker Registry**: Update `.github/workflows/ci-cd.yml` with your registry details
5. **K8s**: Requires `kubectl` and a cluster (minikube/kind/Docker Desktop)

## 🔍 Testing the Pipeline

```bash
# Unit tests
pytest tests/unit/ -v

# API test
curl -X GET http://localhost:8000/health

# Integration test (local)
python deploy/smoke_tests.py

# Code quality
flake8 src/ tests/
black --check src/ tests/
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` in venv |
| CUDA not available | Service auto-uses CPU, check logs |
| Port 8000 in use | Change port in uvicorn or kill process |
| Docker build fails | Ensure model exists: `models/artifacts/model.pt` |
| K8s pod crash | Check resource limits in `deployment.yaml` |

## 📞 Support

- Check README.md for detailed documentation
- Review `.github/copilot-instructions.md` for quick reference
- Look at inline comments in source code

---

**Project Status**: ✅ Complete and Ready for Development

**Next Action**: Download dataset and start training!
