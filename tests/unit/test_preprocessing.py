"""Unit tests for data preprocessing."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

from src.data.preprocessing import DataPreprocessor


@pytest.fixture
def temp_dataset():
    """Create temporary test dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create sample images
        for class_name in ['cats', 'dogs']:
            class_dir = tmpdir_path / class_name
            class_dir.mkdir()
            
            for i in range(5):
                img = Image.new('RGB', (100, 100), color=(73, 109, 137))
                img.save(class_dir / f'sample_{i}.jpg')
        
        yield str(tmpdir_path)


def test_preprocessor_initialization():
    """Test DataPreprocessor initialization."""
    preprocessor = DataPreprocessor(image_size=224, seed=42)
    assert preprocessor.image_size == 224
    assert preprocessor.seed == 42


def test_resize_and_normalize(temp_dataset):
    """Test image resizing and normalization."""
    preprocessor = DataPreprocessor(image_size=224)
    
    # Get a sample image path
    sample_image = list(Path(temp_dataset).glob('*/*.jpg'))[0]
    
    img_array = preprocessor.resize_and_normalize(str(sample_image))
    
    # Check shape
    assert img_array.shape == (224, 224, 3)
    
    # Check normalization (values between 0 and 1)
    assert img_array.min() >= 0
    assert img_array.max() <= 1
    
    # Check dtype
    assert img_array.dtype == np.float32


def test_process_dataset(temp_dataset):
    """Test full dataset preprocessing."""
    with tempfile.TemporaryDirectory() as output_dir:
        preprocessor = DataPreprocessor(image_size=224, seed=42)
        
        stats = preprocessor.process_dataset(temp_dataset, output_dir)
        
        # Check stats
        assert stats['total_samples'] == 10
        assert stats['train_samples'] == 8  # 80%
        assert stats['val_samples'] == 1    # 10%
        assert stats['test_samples'] == 1   # 10%
        assert stats['image_size'] == 224
        assert stats['num_classes'] == 2
        
        # Check output files
        output_path = Path(output_dir)
        assert (output_path / 'train_data' / 'images.npy').exists()
        assert (output_path / 'train_data' / 'labels.npy').exists()
        assert (output_path / 'val_data' / 'images.npy').exists()
        assert (output_path / 'test_data' / 'images.npy').exists()
        assert (output_path / 'stats.json').exists()
        
        # Check data shapes
        train_images = np.load(output_path / 'train_data' / 'images.npy')
        train_labels = np.load(output_path / 'train_data' / 'labels.npy')
        
        assert train_images.shape == (8, 224, 224, 3)
        assert train_labels.shape == (8,)
        assert set(np.unique(train_labels)) == {0, 1}


def test_invalid_image_handling(temp_dataset):
    """Test handling of invalid images."""
    preprocessor = DataPreprocessor(image_size=224)
    
    # Try to process non-existent image
    with pytest.raises(Exception):
        preprocessor.resize_and_normalize('/path/to/nonexistent/image.jpg')
