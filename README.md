# BBoxCut: Targeted Data Augmentation for Occluded Object Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![arXiv](https://img.shields.io/badge/arXiv-2503.24032-b31b1b.svg)](https://arxiv.org/abs/2503.24032)

> **A novel data augmentation technique that improves object detection under occlusions by intelligently masking regions within bounding boxes using adaptive color selection.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance](#-performance)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🎯 Overview

BBoxCut addresses the challenge of detecting objects under occlusion in agricultural settings, specifically wheat head detection in field conditions. The technique simulates realistic occlusions by strategically masking regions within object bounding boxes.

### The Problem

Field conditions present significant detection challenges:
- 🍃 **Leaf occlusions** covering wheat heads
- 🌾 **Overlapping wheat heads** creating partial visibility
- ☀️ **Varying lighting conditions** affecting image quality
- 🏃 **Motion blur** from wind or equipment movement

Traditional augmentation methods apply random masking without considering object locations, often degrading performance when occlusions already exist.

### Our Solution

BBoxCut intelligently:
1. Identifies non-overlapping bounding boxes suitable for masking
2. Estimates dominant colors from the image for realistic occlusion simulation
3. Applies localized masks within selected bounding boxes
4. Preserves already occluded objects to avoid over-augmentation

---

## ✨ Key Features

- **Smart Mask Placement**: IoU-based filtering identifies suitable masking candidates
- **Adaptive Color Selection**: Histogram analysis determines realistic mask colors
- **Probabilistic Control**: Fine-grained parameters control augmentation intensity
- **Architecture Agnostic**: Compatible with various detector architectures
- **Easy Integration**: Works seamlessly with Albumentations library

---

## 📊 Performance

BBoxCut achieves significant improvements on the GWHD 2021 dataset:

| Model | Baseline | CutOut | Region-Aware RE | **BBoxCut (Ours)** | **Improvement** |
|-------|----------|--------|-----------------|-------------------|-----------------| 
| Faster R-CNN | 51.14 | 51.29 | 52.23 | **53.90** | **+2.76** |
| FCOS* | 57.18 | 57.87 | 58.82 | **60.44** | **+3.26** |
| DETR* | 57.40 | 58.20 | 58.50 | **59.30** | **+1.90** |

**Note**: This repository currently contains the Faster R-CNN implementation. FCOS and DETR implementations will be uploaded soon.

### Ablation Study: Mask Color Impact

| Mask Color Strategy | mAP (Faster R-CNN) | Change from Baseline |
|--------------------|--------------------|----------------------|
| Black | 48.75 | -2.39 |
| Gray | 50.51 | -0.63 |
| White | 52.75 | +1.61 |
| Random | 52.70 | +1.56 |
| **Dominant (Ours)** | **53.90** | **+2.76** |

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB RAM minimum

### Step 1: Clone the Repository

```bash
git clone https://github.com/Yasashwini2005/BBoxCut.git
cd BBoxCut
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Required Dependencies

The `requirements.txt` includes:
- PyTorch 2.0+ with CUDA support
- PyTorch Lightning for training
- Albumentations for data augmentation
- OpenCV for image processing
- Standard ML libraries (numpy, pandas, scikit-learn)

---

## 📦 Dataset Setup

This project uses the **Global Wheat Head Detection (GWHD) 2021** dataset.

### Step 1: Download Dataset

1. Visit the [GWHD 2021 Competition Page](https://www.aicrowd.com/challenges/global-wheat-challenge-2021)
2. Register and download the dataset
3. Extract the downloaded files

### Step 2: Organize Dataset Structure

Organize your dataset folder as follows:

```
gwhd_2021/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── competition_train.csv
├── competition_val.csv
└── competition_test.csv
```

### Step 3: Update Dataset Path

Open `fastercnn.ipynb` and update the dataset path:

```python
# Update these paths to point to your dataset location
DATA_PATH = '/path/to/your/gwhd_2021'
TRAIN_CSV = os.path.join(DATA_PATH, 'competition_train.csv')
VAL_CSV = os.path.join(DATA_PATH, 'competition_val.csv')
IMAGE_DIR = os.path.join(DATA_PATH, 'images')
```

### CSV Format

Each CSV file contains:
- `image_name`: Image filename (e.g., "abc123.jpg")
- `BoxesString`: Bounding boxes in format "x y w h; x y w h; ..."
- `domain`: Geographic domain/location identifier

---

## 🚀 Getting Started

### Quick Start Guide

1. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

2. **Open the Training Notebook**

Navigate to `fastercnn.ipynb` in the Jupyter interface.

3. **Configure Training Parameters**

In the notebook, you can adjust:

```python
# Training Configuration
BATCH_SIZE = 2              # Adjust based on GPU memory
LEARNING_RATE = 1e-5       # Learning rate for optimizer
MAX_EPOCHS = 50            # Maximum training epochs
WEIGHT_DECAY = 0.0001      # L2 regularization

# BBoxCut Augmentation Parameters
P_AUG = 0.3               # Probability of applying BBoxCut
P_MASK = 0.3              # Probability of masking each box
ALPHA_W = 0.3             # Max width percentage to mask
ALPHA_H = 0.3             # Max height percentage to mask
IOU_THRESHOLD = 0.5       # IoU threshold for overlap detection
```

4. **Run Training**

Execute the cells in the notebook sequentially. The training process will:
- Load and preprocess the dataset
- Apply BBoxCut augmentation during training
- Save checkpoints to the `checkpoints/` directory
- Log metrics for visualization

5. **Monitor Progress**

Training metrics and validation results will be displayed in the notebook output cells.

### Understanding BBoxCut Parameters

- **P_AUG**: Controls how often BBoxCut is applied to an image (0.3 = 30% of images)
- **P_MASK**: Controls how many non-overlapping boxes get masked (0.3 = 30% of eligible boxes)
- **ALPHA_W/ALPHA_H**: Maximum percentage of box dimensions to mask (0.3 = up to 30%)
- **IOU_THRESHOLD**: Determines what constitutes an "overlapping" box (0.5 = 50% overlap)

### Expected Training Time

- **With GPU (CUDA)**: ~3-4 hours for 50 epochs
- **Without GPU (CPU)**: ~20-24 hours for 50 epochs

---

## 📁 Project Structure

```
BBoxCut/
│
├── README.md                    # This file - Project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
│
├── fastercnn.ipynb             # Main training notebook (Faster R-CNN)
├── wheat_dataset.py            # PyTorch Dataset class for GWHD 2021
├── utils.py                    # Helper functions (collate_fn, etc.)
│
└── docs/                       # Additional documentation
    └── paper.pdf               # Research paper (arXiv:2503.24032)
```

---

## 📈 Results

### Quantitative Performance

BBoxCut demonstrates consistent improvements across different detection architectures:

**Faster R-CNN on GWHD 2021 Test Set:**
- Baseline: 51.14 mAP
- With BBoxCut: 53.90 mAP
- **Improvement: +2.76 mAP (+5.4% relative)**

### Why Dominant Color Matters

Our ablation study shows that adaptive color selection (dominant color) significantly outperforms fixed colors:

- **Black masks**: Often too dark, creating unrealistic occlusions (-2.39 mAP)
- **White masks**: Too bright for agricultural scenes (+1.61 mAP)
- **Random colors**: Inconsistent, sometimes unrealistic (+1.56 mAP)
- **Dominant color**: Matches image context, creates realistic occlusions (**+2.76 mAP**)

---

## 🔬 How BBoxCut Works

### Algorithm Overview

1. **Input**: Image with bounding box annotations
2. **Overlap Detection**: Compute IoU between all box pairs
3. **Candidate Selection**: Identify non-overlapping boxes (IoU < threshold)
4. **Color Estimation**: Extract dominant color from image histogram
5. **Mask Application**: For each candidate box (with probability p_mask):
   - Generate random mask dimensions (up to α_w × α_h of box size)
   - Place mask at random position within box
   - Fill with dominant color
6. **Output**: Augmented image with realistic occlusions

### Key Insight

By avoiding already overlapping boxes, BBoxCut ensures that:
- Occluded objects remain visible for learning
- Only clearly visible objects receive additional occlusion
- The model learns robust features under varying occlusion levels


---

## 🙏 Acknowledgments

- **Research England** (Lincoln Agri-Robotics) for funding support
- **School of Computer Science**, University of Lincoln, UK
- The **GWHD 2021** dataset creators and contributors
- **PyTorch** and **Albumentations** communities for excellent tools

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss your ideas.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Yasashwini Sai Gowri P**
- GitHub: [@Yasashwini2005](https://github.com/Yasashwini2005)
- Project Link: [https://github.com/Yasashwini2005/BBoxCut](https://github.com/Yasashwini2005/BBoxCut)

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---


## 🌟 Star History

If you find this project helpful for your research or applications, please consider giving it a star ⭐!

---

<div align="center">

Made with ❤️ for advancing agricultural AI

**Faster R-CNN Implementation Available Now | FCOS & DETR Coming Soon**

</div>
