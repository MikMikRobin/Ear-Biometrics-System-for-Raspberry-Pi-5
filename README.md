# 🎯 Ear Biometrics System - Complete Package

A comprehensive ear biometrics identification and enrollment system with modern GUI and advanced machine learning models, designed for deployment on Raspberry Pi OS with touchscreen interface.

## 🏗️ Software Architecture

The biometric system comprises two main components: the **backend processing pipeline** for core biometric analysis and the **frontend Graphical User Interface (GUI)** for user interaction and feedback. This modular design runs on **Raspberry Pi OS (64-bit, "Bookworm")** and is primarily developed in **Python 3.11.x**.

### 🔧 Backend Development

The backend pipeline uses publicly available libraries to perform the recognition task:

- **YOLOv12 Model**: Implemented using the Ultralytics library for real-time ear detection
- **EfficientNet**: TensorFlow and TensorFlow Lite libraries for feature extraction
  - TensorFlow for initial model training
  - TensorFlow Lite for optimized on-device inference on Raspberry Pi
- **k-Nearest Neighbors (k-NN)**: Scikit-learn implementation for identity matching and performance metrics
- **OpenCV**: Fundamental image and video manipulation, camera frame capture, and preprocessing

### 🖥️ Frontend Development

The frontend component is a custom GUI application designed to run on the **Pimoroni HyperPixel 4.0 Square Touch screen**:

- **Primary Interface**: Displays live video feed with real-time system status updates
- **Identification Results**: Shows verified identity or "No Match" notification
- **System States**: Active processing, access granted/denied, and system disabled modes
- **Touch-Optimized**: Designed for finger interaction on embedded displays

## 📦 Package Contents

This self-contained package includes everything needed to run the ear biometrics system:

### 🐍 Core Python Files
- `run_ear_biometrics.py` - **Main launcher script** (run this!)
- `ear_biometrics_gui.py` - Modern GUI application with Liquid Glass theme
- `biometric_pipeline.py` - Core biometric processing pipeline
- `model_loader.py` - Universal model loader for robust ML model handling
- `setup.py` - Automated setup and dependency installer

### 🤖 Pre-trained Models
- `best.pt` - YOLO ear detection model
- `excellent_ear_model_best.pth` - Excellent EfficientNet feature extractor
- `efficientnet_b4_ultimate_best.pth` - Ultimate EfficientNet model
- **TIMM Models** - Automatically downloaded (including `efficientnet_retrained_final`)

### 🗃️ Database & Storage
- `ear_biometrics_v3.db` - SQLite database for biometric data
- `enrollment_samples/` - Directory for storing enrollment images

### 📚 Documentation
- `requirements.txt` - Python dependencies
- `USAGE_GUIDE.md` - Detailed usage instructions
- `README.txt` - Additional notes

## 🚀 Quick Start

### 1. Setup (First Time Only)
```bash
cd EarBiometricsPackage
python setup.py
```

### 2. Run the Application
```bash
python run_ear_biometrics.py
```

## ✨ Key Features

### 🎨 Modern GUI
- **Dark Blue Theme** - Beautiful, modern interface
- **Tabbed Interface** - Identification, Enrollment, Database, Options
- **Real-time Video** - Live camera feed with ear detection
- **Visual Guidelines** - Target zones for optimal positioning

### 🧠 Advanced Models
- **Multiple Model Types**: Excellent, Ultimate, TIMM Models
- **Default Model**: `efficientnet_retrained_final` (high performance)
- **Universal Loader**: Handles model compatibility issues automatically
- **GPU Support**: Automatic CUDA detection and acceleration

### 📊 Robust Matching
- **Optimized kNN**: Adaptive k-selection and weighted scoring
- **Multiple Metrics**: Cosine similarity with confidence thresholds
- **Quality Control**: Size validation and confidence filtering

### 🔐 Secure Database
- **SQLite Storage**: Local, secure biometric data storage
- **Migration Support**: Automatic schema updates
- **Model Compatibility**: Tracks feature dimensions and model types

## 🎯 Default Configuration

The system is pre-configured with optimal settings:
- **Model Type**: TIMM Models
- **Default Model**: `efficientnet_retrained_final` 
- **Confidence Threshold**: 90%
- **Match Margin**: 8%
- **Max Enrollment Samples**: 5

## 📋 System Requirements

### Software
- Python 3.8+
- Windows/Linux/macOS
- Webcam or USB camera

### Hardware
- **Minimum**: 4GB RAM, CPU processing
- **Recommended**: 8GB+ RAM, NVIDIA GPU with CUDA support
- **Storage**: ~500MB for models and data

## 🛠️ Dependencies

All dependencies are automatically installed by `setup.py`:
- PyTorch (with CUDA support if available)
- OpenCV for computer vision
- YOLO (Ultralytics) for object detection
- TIMM for pre-trained models
- Tkinter for GUI (usually included with Python)
- scikit-learn for machine learning
- PIL, NumPy, SQLite3

## 📖 Usage Workflow

### 1. **Load Models**
   - Select model type (default: TIMM Models)
   - Choose model (default: efficientnet_retrained_final)
   - Load YOLO detection model
   
### 2. **Enroll Person**
   - Switch to Enrollment tab
   - Enter person's name
   - Start enrollment process
   - Capture 5 samples with different angles
   - Accept and save samples

### 3. **Identify Person**
   - Switch to Identification tab
   - Start camera
   - System automatically detects and identifies ears
   - Results shown in real-time

### 4. **Manage Database**
   - View enrolled persons
   - Check database statistics
   - Delete or export data
   - View sample images

## 🔧 Troubleshooting

### Model Loading Issues
- Run `setup.py` to ensure dependencies are installed
- Check internet connection for TIMM model downloads
- Verify model files are in the package directory

### Camera Issues
- Check camera permissions
- Try different camera indices in settings
- Ensure no other applications are using the camera

### Performance Issues
- Enable GPU acceleration in Options tab
- Adjust confidence thresholds for your use case
- Close other resource-intensive applications

## 🎪 Advanced Features

### Model Flexibility
- **Local TIMM Models**: Automatically detected and marked as "Local"
- **Fallback Handling**: Graceful handling when pretrained weights fail
- **Architecture Detection**: Automatic model type recognition

### Quality Control
- **Visual Guidelines**: Target zones for optimal ear positioning
- **Size Validation**: Minimum and target ear size requirements
- **Confidence Filtering**: Multiple threshold levels

### Performance Optimization
- **Adaptive kNN**: k-selection based on database size
- **Parallel Processing**: Multi-core CPU utilization
- **Memory Management**: Efficient feature storage and retrieval

## 📞 Support

For issues or questions:
1. Check `USAGE_GUIDE.md` for detailed instructions
2. Verify all model files are present
3. Ensure dependencies are properly installed
4. Check camera and GPU drivers

## 🤖 Model Development and Evaluation

### YOLOv12 Fine-Tuning Process

The training procedure for the ear detection model follows a systematic approach:

1. **Dataset Preparation**: 
   - **VGG-FaceEar Dataset**: 90% used for training, 10% for testing
   - **UERC 2019 Dataset**: ~80% used for final testing and validation

2. **Training Process**:
   - Fine-tune YOLOv12 architecture on VGG-FaceEar training data
   - Iterative learning to identify and map intricate ear features
   - Comprehensive two-pronged evaluation:
     - Testing on 10% VGG-FaceEar dataset (familiar distribution)
     - Validation on separate UERC 2019 data (generalization test)

3. **Deployment**:
   - Convert trained model to Raspberry Pi AI Camera compatible format
   - Optimize for on-device inference performance

### EfficientNetLite-4 Fine-tuning Process

The feature extraction model uses EfficientNetLite-4 as the pre-trained base:

1. **Training Approaches**:
   - **Full Fine-tuning**: All layers adjust weights based on dataset
   - **Layer-wise Fine-tuning**: First 3 layers frozen, middle and final layers adjustable

2. **Evaluation Phases**:
   - **Baseline Classification**: Test as closed-set classifier with classification layer
   - **Feature Extraction**: Remove classification layer, pair with k-NN classifier
   - **Cross-Dataset Generalization**: Test on UERC 2019 subset without retraining

3. **Deployment Optimization**:
   - Convert to Hailo 8 compatible format (Raspberry Pi AI HAT+)
   - Optimize for on-device inference with TensorFlow Lite

## 🏆 Model Performance

Default model rankings (accuracy on test datasets):
- 🟢 **efficientnet_retrained_final**: 95.61% (Recommended)
- 🟢 efficientnetv2_m: 85.2%
- 🟢 regnetx_32gf: 83.7%
- 🟡 efficientnet_b7: 78.1%
- 🟡 efficientnet_b6: 81.8%

## 🎯 System Requirements

### Target Platform
- **Operating System**: Raspberry Pi OS (64-bit, "Bookworm")
- **Python Version**: 3.11.x
- **Display**: Pimoroni HyperPixel 4.0 Square Touch screen
- **Camera**: Raspberry Pi AI Camera or compatible USB camera

### Hardware Recommendations
- **Minimum**: Raspberry Pi 4B with 4GB RAM
- **Recommended**: Raspberry Pi 5 with 8GB RAM
- **Storage**: 16GB+ microSD card (Class 10 or better)
- **AI Acceleration**: Raspberry Pi AI HAT+ with Hailo 8 processor (optional)

---

**🎉 Ready to use! Run `python run_ear_biometrics.py` to start.**

