"""CNN model for cats vs dogs classification."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class SimpleCNN(nn.Module):
    """Simple CNN for binary image classification."""
    
    def __init__(self, num_classes: int = 2):
        """Initialize Simple CNN.
        
        Args:
            num_classes: Number of classes (default: 2 for cats vs dogs).
        """
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, height, width).
            
        Returns:
            Output tensor (batch_size, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet50BinaryClassifier(nn.Module):
    """ResNet50 fine-tuned for binary image classification."""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """Initialize ResNet50 classifier.
        
        Args:
            num_classes: Number of classes (default: 2).
            pretrained: Whether to use pretrained weights.
        """
        super(ResNet50BinaryClassifier, self).__init__()
        
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # Replace final classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.resnet(x)


def create_model(model_type: str = 'simple_cnn', num_classes: int = 2, **kwargs) -> nn.Module:
    """Factory function to create model.
    
    Args:
        model_type: Type of model ('simple_cnn' or 'resnet50').
        num_classes: Number of classes.
        **kwargs: Additional arguments for model.
        
    Returns:
        Initialized model.
    """
    if model_type == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_type == 'resnet50':
        return ResNet50BinaryClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
