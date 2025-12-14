# MobileNetV2 CIFAR-10 Training and Testing

This repository contains implementations for training and testing MobileNetV2 on the CIFAR-10 dataset, including both baseline FP32 training and quantization-aware training (QAT) for model compression.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Baseline Model](#training-baseline-model)
- [Testing Models](#testing-models)
- [Model Compression with Quantization](#model-compression-with-quantization)
- [Results and Visualization](#results-and-visualization)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster training)
- At least 4GB of free disk space

### Required Packages
```bash
pip install torch torchvision matplotlib pandas numpy
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## Project Structure

```
pytorch-cifar/
├── README.md                           # This file
├── train_baseline.py                   # Standalone baseline training script
├── test_baseline.py                    # Test script for baseline model
├── test_retrained.py                   # Test script for QAT model
├── mobilenetv2_cifar10_colab.ipynb    # Complete notebook with quantization
├── models/                             # Model architecture definitions
├── data/                               # CIFAR-10 dataset (auto-downloaded)
├── best_baseline_model.pth             # Trained baseline model weights
├── best_model_qat_retrained.pth        # Trained QAT model weights
└── baseline_training_curves.png        # Training visualization
```

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd pytorch-cifar
pip install torch torchvision matplotlib pandas numpy
```

### 2. Train Baseline Model (FP32)
```bash
python train_baseline.py
```

### 3. Test Baseline Model
```bash
python test_baseline.py
```

### 4. View Results
Check the generated `baseline_training_curves.png` for training progress visualization.

## Training Baseline Model

### Using the Standalone Script

The `train_baseline.py` script provides a complete training pipeline for the baseline MobileNetV2 model:

```bash
python train_baseline.py
```

**What this script does:**
- Downloads CIFAR-10 dataset automatically (if not present)
- Initializes MobileNetV2 model for CIFAR-10 (10 classes)
- Trains for 50 epochs with data augmentation
- Saves the best model based on test accuracy
- Generates training curves visualization

**Training Configuration:**
- **Architecture:** MobileNetV2 (from kuangliu/pytorch-cifar)
- **Dataset:** CIFAR-10 (32×32 RGB images, 10 classes)
- **Batch Size:** 128 (training), 100 (testing)
- **Epochs:** 50
- **Optimizer:** SGD with momentum (0.9) and weight decay (5e-4)
- **Learning Rate:** 0.1 with Cosine Annealing scheduler
- **Data Augmentation:** Random crop, horizontal flip, normalization

**Expected Output:**
```
================================================================================
BASELINE MODEL TRAINING (FP32)
================================================================================
Using device: cuda

Preparing CIFAR-10 dataset...
Training set: 50000 images
Test set: 10000 images

Initializing MobileNetV2 model for FP32 baseline...
Total trainable parameters: 2,296,922
Model size (FP32): 8.79 MB

Starting training for 50 epochs...
--------------------------------------------------------------------------------
Epoch 1 | Loss: 1.523 | Acc: 45.234%
Test Loss: 1.234 | Test Acc: 56.780%
✓ Saved best model with test accuracy: 56.780% at epoch 1
...
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.