.PHONY: help install test run-api run-training docker-build docker-run deploy-compose deploy-k8s smoke-tests clean

help:
	@echo "MLOps Project - Available Commands"
	@echo "=================================="
	@echo "install       - Install dependencies"
	@echo "test          - Run unit tests with coverage"
	@echo "test-quick    - Run tests without coverage"
	@echo "lint          - Run code quality checks"
	@echo "format        - Format code with black"
	@echo "preprocess    - Run data preprocessing"
	@echo "train         - Train model with MLflow"
	@echo "run-api       - Start FastAPI server"
	@echo "docker-build  - Build Docker image"
	@echo "docker-run    - Run Docker container"
	@echo "deploy-compose - Deploy with Docker Compose"
	@echo "deploy-k8s    - Deploy to Kubernetes"
	@echo "smoke-tests   - Run post-deployment tests"
	@echo "clean         - Clean artifacts"

install:
	pip install -r requirements.txt

test:
	pytest tests/unit/ -v --cov=src --cov-report=html

test-quick:
	pytest tests/unit/ -v

lint:
	flake8 src/ tests/ --count --select=E9,F63,F7,F82
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

preprocess:
	python src/data/preprocessing.py

train:
	python src/models/train.py

mlflow-ui:
	mlflow ui

run-api:
	uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t cats-dogs-classifier:latest .

docker-run:
	docker run -p 8000:8000 \
		-v $(PWD)/models/artifacts:/app/models/artifacts \
		cats-dogs-classifier:latest

deploy-compose:
	docker-compose -f deploy/docker-compose/docker-compose.yml up -d

deploy-compose-down:
	docker-compose -f deploy/docker-compose/docker-compose.yml down

deploy-k8s:
	kubectl apply -f deploy/k8s/deployment.yaml

smoke-tests:
	python deploy/smoke_tests.py --url http://localhost:8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
