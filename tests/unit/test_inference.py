"""Unit tests for inference service."""

import pytest
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from src.inference.api import app, ModelInference


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    return img


@pytest.fixture
def sample_image_base64(sample_image):
    """Create base64 encoded sample image."""
    buffer = BytesIO()
    sample_image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert isinstance(data['model_loaded'], bool)
    assert data['version'] == '1.0.0'


def test_info_endpoint(client):
    """Test info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data['name'] == 'Cats vs Dogs Classification API'
    assert data['classes'] == ['cat', 'dog']
    assert data['image_size'] == 224


def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert 'total_requests' in data
    assert 'total_predictions' in data


def test_predict_endpoint(client, sample_image_base64):
    """Test predict endpoint with base64 image."""
    response = client.post(
        "/predict",
        json={"image": sample_image_base64}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert 'prediction' in data
    assert data['prediction'] in ['cat', 'dog']
    assert 'confidence' in data
    assert isinstance(data['confidence'], float)
    assert 0 <= data['confidence'] <= 1
    assert 'probabilities' in data
    assert 'cat' in data['probabilities']
    assert 'dog' in data['probabilities']


def test_predict_file_endpoint(client, sample_image):
    """Test predict-file endpoint."""
    buffer = BytesIO()
    sample_image.save(buffer, format='JPEG')
    buffer.seek(0)
    
    response = client.post(
        "/predict-file",
        files={"file": ("test.jpg", buffer, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert 'prediction' in data
    assert data['prediction'] in ['cat', 'dog']
    assert 'confidence' in data


def test_invalid_base64_image(client):
    """Test error handling for invalid base64."""
    response = client.post(
        "/predict",
        json={"image": "invalid_base64_data"}
    )
    assert response.status_code == 400


def test_model_inference_initialization():
    """Test ModelInference initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inference = ModelInference(model_path='nonexistent.pt', device='cpu')
        assert inference.device == 'cpu'
        assert inference.image_size == 224
        assert inference.class_names == ['cat', 'dog']


def test_model_inference_preprocess():
    """Test image preprocessing in ModelInference."""
    inference = ModelInference(device='cpu')
    
    img = Image.new('RGB', (100, 100), color=(75, 125, 175))
    tensor = inference.preprocess_image(img)
    
    # Check shape (batch_size, channels, height, width)
    assert tensor.shape == (1, 3, 224, 224)
    
    # Check normalization
    assert tensor.min() >= 0
    assert tensor.max() <= 1


def test_model_inference_predict():
    """Test prediction with ModelInference."""
    inference = ModelInference(device='cpu')
    
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    result = inference.predict(img)
    
    assert 'prediction' in result
    assert result['prediction'] in ['cat', 'dog']
    assert 'confidence' in result
    assert 'probabilities' in result
    
    # Probabilities should sum to approximately 1
    total_prob = sum(result['probabilities'].values())
    assert 0.99 <= total_prob <= 1.01
