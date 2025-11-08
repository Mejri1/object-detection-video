# Real-Time Object Detection System using YOLO and GPU Acceleration

A computer vision project implementing real-time object detection with YOLO, fine-tuned for specific car recognition. The system leverages NVIDIA GPU acceleration for optimal performance.

## Project Overview

This project was developed as a computer vision assignment requiring real-time object detection using YOLO and GPU acceleration. While the initial goal was to implement a general object detection system, we extended it by fine-tuning the model to recognize a specific car (Omar's car) among many vehicles.

## Features

- ✅ CUDA-enabled GPU acceleration
- ✅ Fine-tuned YOLOv8 Large model for car recognition
- ✅ Two-stage detection pipeline (general objects + car classification)
- ✅ Real-time inference capability (~20 FPS)
- ✅ Comprehensive data preprocessing with Label Studio
- ✅ Extensive data augmentation
- ✅ High accuracy (98.8% mAP@0.5)

## Project Structure

```
object_detection_project/
├── data/                   # Dataset (images, labels, configs)
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_train_omar_car.ipynb
│   └── 04_predict_general_and_car_classification.ipynb
├── models/                 # Model weights (gitignored)
├── runs/                   # Training outputs (gitignored)
├── src/                    # Source code
├── deployment/             # Deployment scripts
└── report.tex             # LaTeX report
```

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 3060)
- Minimum 6GB GPU memory
- CUDA 12.6+

### Software
- Python 3.8+
- PyTorch with CUDA support
- Ultralytics YOLO
- Label Studio (for annotation)

## Installation

### 1. CUDA Setup
1. Download and install [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads)
2. Download and install [cuDNN](https://developer.nvidia.com/cudnn)
3. Add CUDA to system PATH:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
   ```

### 2. Python Environment
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install ultralytics label-studio numpy pandas matplotlib pillow
```

### 3. Verify GPU
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
```

## Usage

### 1. Data Preparation
- Use Label Studio to annotate images
- Export labels in YOLO format
- Organize data in `data/images/` and `data/labels/`

### 2. Exploratory Data Analysis
```bash
jupyter notebook notebooks/02_eda.ipynb
```

### 3. Training
```bash
jupyter notebook notebooks/03_train_omar_car.ipynb
```
The notebook will:
- Automatically detect GPU and adjust batch size
- Perform two-phase training (frozen + unfrozen)
- Save best model to `runs/detect/omar_car_training_large/weights/best.pt`

### 4. Prediction
```bash
jupyter notebook notebooks/04_predict_general_and_car_classification.ipynb
```
This notebook:
- Detects all objects using YOLOv8n
- Classifies cars using the fine-tuned model
- Visualizes results with color-coded bounding boxes

## Results

### Model Performance
- **Precision**: 99.8%
- **Recall**: 97.2%
- **mAP@0.5**: 98.8%
- **mAP@0.5:0.95**: 80.9%

### Inference Speed
- **GPU**: ~50ms per image
- **Real-time**: ~20 FPS
- **Batch processing**: Supported

## Report

A comprehensive LaTeX report is available in `report.tex`, covering:
- CUDA configuration and setup
- Data preprocessing with Label Studio
- Exploratory data analysis
- Model architecture and training
- GPU performance analysis
- Results and discussion

To compile the report:
```bash
pdflatex report.tex
```

## License

This project is for educational purposes.

## Authors

- Omar Mejri
- [Partner Name]

## Acknowledgments

Special thanks to the professor for guidance and support throughout the project.
