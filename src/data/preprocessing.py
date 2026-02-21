"""Data preprocessing module for cats vs dogs classification."""

import os
import json
import shutil
from pathlib import Path
from typing import Tuple, List
import logging

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataPreprocessor:
    """Preprocess cats and dogs dataset."""
    
    def __init__(self, image_size: int = 224, seed: int = 42):
        """Initialize preprocessor.
        
        Args:
            image_size: Target size for images (square).
            seed: Random seed for reproducibility.
        """
        self.image_size = image_size
        self.seed = seed
        np.random.seed(seed)
    
    def resize_and_normalize(self, image_path: str) -> np.ndarray:
        """Resize image to target size and convert to RGB.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Normalized image array (0-1 range).
            
        Raises:
            Exception: If image cannot be loaded or processed.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise
    
    def process_dataset(
        self,
        raw_data_path: str,
        output_path: str
    ) -> dict:
        """Process entire dataset and split into train/val/test.
        
        Args:
            raw_data_path: Path to raw dataset directory.
            output_path: Path to save processed data.
            
        Returns:
            Dictionary with split statistics.
        """
        raw_path = Path(raw_data_path)
        output_base = Path(output_path)
        output_base.mkdir(parents=True, exist_ok=True)
        
        images = []
        labels = []
        
        # Collect images and labels
        for class_name in ['cats', 'dogs']:
            class_path = raw_path / class_name
            if not class_path.exists():
                logger.warning(f"Class directory not found: {class_path}")
                continue
            
            label = 0 if class_name == 'cats' else 1
            for image_file in class_path.glob('*.jpg'):
                try:
                    img_array = self.resize_and_normalize(str(image_file))
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    logger.warning(f"Skipped {image_file}: {str(e)}")
        
        if not images:
            raise ValueError(f"No images found in {raw_data_path}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images")
        
        # Split data: 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.2, random_state=self.seed, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.seed, stratify=y_temp
        )
        
        # Save splits
        for split_name, (X, y) in [
            ('train_data', (X_train, y_train)),
            ('val_data', (X_val, y_val)),
            ('test_data', (X_test, y_test))
        ]:
            split_path = output_base / split_name
            split_path.mkdir(parents=True, exist_ok=True)
            
            np.save(split_path / 'images.npy', X)
            np.save(split_path / 'labels.npy', y)
            
            logger.info(f"{split_name}: {len(X)} samples")
        
        stats = {
            'total_samples': len(images),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'image_size': self.image_size,
            'num_classes': 2,
            'class_names': ['cats', 'dogs']
        }
        
        with open(output_base / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats


def main():
    """Main preprocessing function."""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    preprocessor = DataPreprocessor(
        image_size=params['data']['image_size'],
        seed=params['environment']['seed']
    )
    
    stats = preprocessor.process_dataset(
        raw_data_path='data/raw',
        output_path='data/processed'
    )
    
    logger.info(f"Preprocessing complete. Stats: {stats}")


if __name__ == '__main__':
    main()
