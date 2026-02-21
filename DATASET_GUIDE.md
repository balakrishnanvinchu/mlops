# Dataset Setup Guide

## Download Cats vs Dogs Dataset

This guide helps you download and set up the Cats vs Dogs dataset for the MLOps project.

## Option 1: Download from Kaggle (Recommended)

### Prerequisites
- Kaggle account (free)
- Kaggle CLI installed

### Steps

1. **Install Kaggle CLI**:
```bash
pip install kaggle
```

2. **Get Kaggle API Credentials**:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New Token"
   - This downloads `kaggle.json`
   - Place it in `~/.kaggle/kaggle.json` (create if needed)

3. **Download Dataset**:
```bash
cd mlops-cats-dogs
kaggle datasets download -d shaunlarsen/cat-and-dog-images
unzip cat-and-dog-images.zip -d data/raw/
```

4. **Verify Structure**:
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

## Option 2: Download Manually

1. Visit: https://www.kaggle.com/datasets/shaunlarsen/cat-and-dog-images
2. Click "Download"
3. Extract to `data/raw/`
4. Rename folders if needed to match structure above

## Option 3: Use Microsoft Cats vs Dogs Dataset

```bash
cd data/raw
# Create directories
mkdir cats dogs

# Download using curl (example)
# Download from https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
# Extract and organize into cats/ and dogs/ folders
```

## Verify Dataset

After downloading, verify the structure:

```bash
python -c "
from pathlib import Path
raw_path = Path('data/raw')
cats = len(list((raw_path / 'cats').glob('*.jpg')))
dogs = len(list((raw_path / 'dogs').glob('*.jpg')))
print(f'Cats: {cats}')
print(f'Dogs: {dogs}')
print(f'Total: {cats + dogs}')
"
```

## Expected Statistics

- Total images: 5,000+ (approx)
- Cats: ~2,500 images
- Dogs: ~2,500 images

## Next Steps

Once dataset is ready:

```bash
# Preprocess data
python src/data/preprocessing.py

# Verify preprocessing
python -c "
import json
with open('data/processed/stats.json') as f:
    stats = json.load(f)
    for k, v in stats.items():
        print(f'{k}: {v}')
"

# Train model
python src/models/train.py
```

## Troubleshooting

### No images found error
- Check folder structure matches: `data/raw/cats/` and `data/raw/dogs/`
- Ensure .jpg files are in the directories
- Use absolute paths if needed

### Download fails
- Check internet connection
- Try downloading from browser directly
- Check Kaggle credentials if using CLI

### Memory issues
- Dataset is ~1-2 GB uncompressed
- Ensure 3 GB free space
- Can reduce image count for testing

## Alternative: Smaller Test Dataset

For testing without full dataset:

```bash
# Create test structure
mkdir -p data/raw/cats data/raw/dogs

# Add a few test images (create with PIL if needed)
python -c "
from PIL import Image
import os

# Create 10 test cat images
for i in range(10):
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    img.save(f'data/raw/cats/test_cat_{i}.jpg')

# Create 10 test dog images
for i in range(10):
    img = Image.new('RGB', (224, 224), color=(150, 100, 50))
    img.save(f'data/raw/dogs/test_dog_{i}.jpg')

print('Test dataset created with 20 images')
"
```

## Docker-based Download

If you prefer isolation:

```bash
docker run --rm -it -v $PWD/data:/data \
  -e KAGGLE_USERNAME=your_username \
  -e KAGGLE_KEY=your_api_key \
  python:3.10 bash \
  -c "pip install kaggle && kaggle datasets download -d shaunlarsen/cat-and-dog-images -p /data/raw/ && unzip /data/raw/*.zip -d /data/raw/"
```

---

**Once dataset is ready, proceed with model training!**
