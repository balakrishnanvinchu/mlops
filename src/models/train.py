"""Model training script with MLflow integration."""

import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import yaml
import mlflow
import mlflow.pytorch

from src.models.cnn_model import create_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    """Trainer for CNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model.
            device: Device to train on ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.device = device
        self.kwargs = kwargs
    
    def train_epoch(
        self,
        train_loader,
        criterion,
        optimizer,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            epoch: Current epoch number.
            
        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx + 1}: "
                    f"Loss = {loss.item():.4f}"
                )
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(
        self,
        val_loader,
        criterion
    ) -> Tuple[float, float, dict]:
        """Validate model.
        
        Args:
            val_loader: Validation data loader.
            criterion: Loss function.
            
        Returns:
            Tuple of (loss, accuracy, metrics dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        return epoch_loss, accuracy, metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 20,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> dict:
        """Train model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs.
            learning_rate: Learning rate.
            weight_decay: Weight decay for optimizer.
            
        Returns:
            Training history dictionary.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_metrics': []
        }
        
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, epoch + 1
            )
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_metrics'].append(val_metrics)
            
            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }, step=epoch)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return history


def load_data_loaders(
    data_path: str,
    batch_size: int = 64
):
    """Load data loaders from preprocessed data.
    
    Args:
        data_path: Path to preprocessed data.
        batch_size: Batch size for data loaders.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    data_base = Path(data_path)
    
    datasets = {}
    for split in ['train_data', 'val_data', 'test_data']:
        split_path = data_base / split
        images = np.load(split_path / 'images.npy')
        labels = np.load(split_path / 'labels.npy')
        
        # Convert to torch tensors (images: B, H, W, C -> B, C, H, W)
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        labels_tensor = torch.from_numpy(labels).long()
        
        dataset = TensorDataset(images_tensor, labels_tensor)
        datasets[split] = dataset
    
    train_loader = DataLoader(datasets['train_data'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val_data'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets['test_data'], batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # MLflow setup
    mlflow.set_experiment('cats-vs-dogs-classification')
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model_type': 'simple_cnn',
            'epochs': params['model']['epochs'],
            'batch_size': params['model']['batch_size'],
            'learning_rate': params['model']['learning_rate'],
            'weight_decay': params['model']['weight_decay']
        })
        
        # Load data
        logger.info("Loading data...")
        train_loader, val_loader, test_loader = load_data_loaders(
            'data/processed',
            batch_size=params['model']['batch_size']
        )
        
        # Create model
        device = params['environment']['device']
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            logger.warning("CUDA not available, using CPU")
        
        model = create_model('simple_cnn', num_classes=2)
        
        # Train
        trainer = ModelTrainer(model, device=device)
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=params['model']['epochs'],
            learning_rate=params['model']['learning_rate'],
            weight_decay=params['model']['weight_decay']
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_metrics = trainer.validate(test_loader, nn.CrossEntropyLoss())
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Metrics: {test_metrics}")
        
        # Log metrics
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_acc': test_acc,
            **{f'test_{k}': v for k, v in test_metrics.items()}
        })
        
        # Save model
        model_path = Path('models/artifacts')
        model_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), model_path / 'model.pt')
        mlflow.pytorch.log_model(model, 'model')
        
        # Save metrics
        metrics = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            **test_metrics,
            'history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_acc': [float(x) for x in history['train_acc']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_acc': [float(x) for x in history['val_acc']]
            }
        }
        
        with open(model_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training complete!")


if __name__ == '__main__':
    main()
