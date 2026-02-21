"""FastAPI inference service for cats vs dogs classification."""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import yaml

from src.models.cnn_model import create_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Request/Response models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class PredictionRequest(BaseModel):
    """Prediction request with base64 encoded image."""
    image: str  # base64 encoded


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: str  # 'cat' or 'dog'
    confidence: float
    probabilities: Dict[str, float]


class ModelInference:
    """Model inference handler."""
    
    def __init__(self, model_path: str = 'models/artifacts/model.pt', device: str = 'cpu'):
        """Initialize inference handler.
        
        Args:
            model_path: Path to saved model weights.
            device: Device for inference ('cpu' or 'cuda').
        """
        self.device = device
        self.model = create_model('simple_cnn', num_classes=2)
        self.image_size = 224
        self.class_names = ['cat', 'dog']
        
        # Load model weights
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using untrained model")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Preprocessed torch tensor.
        """
        # Resize and convert to RGB
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict class for image.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Dictionary with prediction, confidence, and probabilities.
        """
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        pred_class = torch.argmax(probabilities).item()
        confidence = probabilities[pred_class].item()
        
        return {
            'prediction': self.class_names[pred_class],
            'confidence': float(confidence),
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.class_names, probabilities.cpu().numpy())
            }
        }


# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="Binary image classification service for cat and dog images",
    version="1.0.0"
)

# Global inference handler
inference_handler: Optional[ModelInference] = None

# Health check tracking
request_count = 0
prediction_count = 0


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global inference_handler
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_handler = ModelInference(device=device)
    logger.info(f"Inference handler initialized on {device}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    global request_count
    request_count += 1
    
    logger.info(f"Health check request #{request_count}")
    
    return HealthResponse(
        status="healthy",
        model_loaded=inference_handler is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict class for provided image.
    
    Args:
        request: Request with base64 encoded image.
        
    Returns:
        Prediction response.
        
    Raises:
        HTTPException: If image is invalid or processing fails.
    """
    global prediction_count
    prediction_count += 1
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_data))
        
        # Run inference
        result = inference_handler.predict(image)
        
        logger.info(
            f"Prediction #{prediction_count}: {result['prediction']} "
            f"(confidence: {result['confidence']:.4f})"
        )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict class for uploaded file.
    
    Args:
        file: Uploaded image file.
        
    Returns:
        Prediction response.
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        result = inference_handler.predict(image)
        
        logger.info(f"File prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
        
        return result
    
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
async def metrics() -> Dict[str, int]:
    """Get inference metrics."""
    return {
        'total_requests': request_count,
        'total_predictions': prediction_count
    }


@app.get("/info")
async def info() -> Dict[str, Any]:
    """Get service info."""
    return {
        'name': 'Cats vs Dogs Classification API',
        'version': '1.0.0',
        'classes': ['cat', 'dog'],
        'image_size': 224,
        'device': inference_handler.device if inference_handler else 'unknown'
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
