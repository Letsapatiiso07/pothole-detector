# 🚗 Pothole & Road Anomaly Detector

A production-ready computer vision system for real-time pothole and road anomaly detection using deep learning. Specifically enhanced with synthetic data generation to handle challenging African road conditions including dust, rain, mud, and harsh sunlight.

## 🌟 Key Features

### 🎯 Core Capabilities
- **Transfer Learning**: Leverage pre-trained ResNet50, EfficientNet, or MobileNetV2
- **Grad-CAM Visualization**: Interpretable AI showing exactly where potholes are detected
- **Multi-Dataset Support**: Easy integration and merging of multiple pothole datasets

### 🌍 African Road Condition Simulation
- **Dust Effects**: Brownish haze and airborne particles common on unpaved roads
- **Rain Simulation**: Wet surfaces, water reflections, and rain streaks
- **Mud Splatter**: Realistic mud spots and droplets on camera lens
- **Harsh Sunlight**: Overexposure, lens flare, and extreme brightness
- **Dynamic Shadows**: Irregular patterns from trees, buildings, and power lines
- **Worn Surfaces**: Aged road texture with reduced saturation

### 📊 Comprehensive Analytics
- **Accuracy, Precision, Recall, F1-Score**: Complete performance metrics
- **AUC-ROC Curve**: Threshold-independent evaluation
- **Confusion Matrix**: Visual performance breakdown
- **Training Curves**: Real-time monitoring via TensorBoard
- **Classification Reports**: Detailed per-class statistics

## 🚀 Quick Start

### Dataset Setup

**Option 1: Download Kaggle Dataset**
```bash
# Download from Kaggle
kaggle datasets download -d atulyakumar98/pothole-detection-dataset

# Extract
unzip pothole-detection-dataset.zip -d data/

# Verify structure
python dataset_checker.py --data_dir data/pothole-detection-dataset
```

**Option 2: Create Sample Dataset**
```bash
# Generate synthetic test data
python create_sample_data.py --output_dir data/pothole-detection-dataset --num_samples 100
```

### Training

```bash
# Basic training
python main.py --mode train \
    --data_dir data/pothole-detection-dataset \
    --epochs 20 \
    --batch_size 32

# Advanced training with custom parameters
python main.py --mode train \
    --data_dir data/pothole-detection-dataset \
    --model_path best_model.h5 \
    --epochs 50 \
    --batch_size 16 \
    --img_size 224
```

## 🏗️ Architecture

### Model Pipeline

```
Input Image (224x224x3)
    ↓
ResNet50 Base (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense(512) + BatchNorm + Dropout(0.5)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(1, sigmoid) → Pothole Probability
```

### Data Processing Flow

```
Raw Dataset
    ↓
[Data Splitting: 70% Train / 15% Val / 15% Test]
    ↓
[Augmentation: Rotation, Flip, Brightness, Zoom]
    ↓
[African Conditions: Dust, Rain, Mud, Sunlight]
    ↓
Training Pipeline
    ↓
Model Evaluation
    ↓
Real-time Inference
```

## 🔬 Synthetic Data Generation

### Why Synthetic Data?

African roads present unique challenges:
- Unpaved surfaces with heavy dust
- Sudden weather changes (heavy rain)
- Extreme lighting conditions
- Limited labeled datasets for these conditions

### Generate Synthetic Data

```bash
# Generate 500 synthetic images per class
python main.py --mode generate_synthetic \
    --data_dir data/pothole-detection-dataset \
    --synthetic_output data/synthetic \
    --num_synthetic 500

# Merge with original dataset
cp -r data/synthetic/normal/* data/pothole-detection-dataset/normal/
cp -r data/synthetic/pothole/* data/pothole-detection-dataset/pothole/
```

### Available Effects

```python
from synthetic_data import (
    add_dust_effect,      # Dust haze and particles
    add_rain_effect,      # Wet roads and rain streaks
    add_mud_splatter,     # Mud spots and droplets
    add_harsh_sunlight,   # Overexposure and lens flare
    add_shadows,          # Irregular shadow patterns
    add_worn_surface      # Aged road texture
)

import cv2
img = cv2.imread('road_image.jpg')
augmented = add_dust_effect(img)
```

## 📈 Performance Optimization

### For Accuracy

1. **Merge Multiple Datasets**: Combine 3-5 pothole datasets
   ```bash
   python dataset_integration.py --action merge \
       --inputs dataset1/ dataset2/ dataset3/ \
       --output data/merged_dataset/
   ```

2. **Generate Synthetic Data**: Create 1000+ augmented images
3. **Fine-tune More Layers**: Edit `model.py` to unfreeze more layers
4. **Ensemble Models**: Train 3-5 models and average predictions
5. **Adjust Confidence Threshold**: Tune from default 0.7

## 🛠️ Advanced Features

### Multi-Dataset Integration

```bash
# Convert YOLO dataset to binary classification
python dataset_integration.py --action convert_yolo \
    --input yolo_dataset/ \
    --output data/yolo_converted/

# Convert COCO dataset
python dataset_integration.py --action convert_coco \
    --input "annotations.json,images/" \
    --output data/coco_converted/

# Merge all datasets
python dataset_integration.py --action merge \
    --inputs data/kaggle/ data/yolo_converted/ data/coco_converted/ \
    --output data/merged_dataset/

# Balance dataset (handle class imbalance)
python dataset_integration.py --action balance \
    --input data/merged_dataset/ \
    --method oversample
```

## 📁 Project Structure

```
pothole-detector/
├── main.py                      # Main entry point
├── data.py                      # Data loading & augmentation (FIXED)
├── model.py                     # Model architecture
├── train.py                     # Training & evaluation
├── infer.py                     # Real-time inference with Grad-CAM
├── utils.py                     # Visualization utilities
├── synthetic_data.py            # Synthetic data generation
├── dataset_integration.py       # Multi-dataset tools
├── create_sample_data.py        # Sample dataset generator
├── dataset_checker.py           # Dataset structure validator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/                        # Datasets directory
│   ├── pothole-detection-dataset/
│   ├── synthetic/
├── logs/                        # TensorBoard logs
├── pothole_model.h5            # Trained model
├── .gitignore

```

## 🐛 Troubleshooting

### Issue: "Found 0 images belonging to 2 classes"

**Solution:**
```bash
# Run diagnostic
python dataset_checker.py --data_dir data/pothole-detection-dataset --fix

# If empty split directories exist, they'll be removed automatically
```

### Issue: Low Accuracy (<85%)

**Solutions:**
1. Generate more synthetic data (1000+ images)
2. Merge additional datasets
3. Increase training epochs (50-100)
4. Fine-tune more layers
5. Use data balancing

### Issue: Slow Inference

**Solutions:**
1. Use GPU (`pip install tensorflow-gpu`)
2. Switch to MobileNetV2 backbone
3. Reduce image size (`--img_size 160`)
4. Convert to TensorFlow Lite
5. Use model pruning

### Issue: Grad-CAM Not Working

**Solution:** Already fixed in latest `utils.py`. Ensure you're using the updated version.

## 🎯 Results

### Expected Performance

- **Accuracy**: 92-96%
- **Precision**: 90-95%
- **Recall**: 88-93%
- **F1-Score**: 90-94%
- **AUC-ROC**: 0.95-0.98
- **Inference Speed**: 20-50ms per frame (GPU)

### Sample Output

```
=== Test Set Performance ===
Accuracy:  0.9450
Precision: 0.9280
Recall:    0.9150
F1-Score:  0.9215

=== Classification Report ===
              precision    recall  f1-score   support

      Normal       0.95      0.97      0.96       143
     Pothole       0.93      0.92      0.92       140

    accuracy                           0.95       283
   macro avg       0.94      0.94      0.94       283
weighted avg       0.94      0.95      0.94       283
```

## 🚀 Future Enhancements

- [ ] **YOLO Integration**: Full object detection with bounding boxes
- [ ] **Mobile App**: Flutter/React Native deployment
- [ ] **Cloud API**: RESTful API for remote inference
- [ ] **Multi-Class Detection**: Severity levels (minor, moderate, severe)
- [ ] **GPS Integration**: Real-time pothole mapping
- [ ] **Edge Deployment**: Raspberry Pi / NVIDIA Jetson
- [ ] **Active Learning**: Continuous model improvement from user feedback

## 📝 Citation

If you use this project in your research or application:

```bibtex
@software{pothole_detector_2024,
  title = {Pothole & Road Anomaly Detector with African Road Condition Simulation},
  author = {Letsapa, Tiiso},
  year = {2025},
  note = {Deep learning system for real-time pothole detection with synthetic data generation}
}
```

## 📄 License

MIT License - Free for research and commercial use.

## 👤 Author

**Tiiso Letsapa**
- GitHub: [@Letsapatiiso07](https://github.com/Letsapatiiso07)
- LinkedIn: [Tiiso Letsapa](https://linkedin.com/in/tiiso-letsapa-664990209)
- Email: Letsapamyron07@gmail.com

## 🙏 Acknowledgments

- [Kaggle Pothole Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)
- TensorFlow & Keras teams
- OpenCV community
- ResNet, EfficientNet, MobileNet authors
- African AI community for inspiring local-context solutions


**Built with ❤️ for African infrastructure monitoring**