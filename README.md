# BBoxCut: Targeted Data Augmentation for Occluded Object Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2503.24032-b31b1b.svg)](https://arxiv.org/abs/2503.24032)

> **A novel data augmentation technique that uses random localized masking to simulate realistic occlusions, improving wheat head detection under challenging field conditions.**

<div align="center">
  <img src="assets/bboxcut_demo.png" alt="BBoxCut Demo" width="800"/>
  <p><i>BBoxCut selectively masks portions of bounding boxes to simulate real-world occlusions</i></p>
</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage with Albumentations](#usage-with-albumentations)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

BBoxCut addresses the critical challenge of detecting wheat heads under occlusion scenarios in agricultural settings. Unlike generic masking techniques (e.g., Cutout, Random Erasing), BBoxCut:

- **Intelligently identifies** non-overlapping bounding boxes to avoid masking already occluded objects
- **Adapts mask color** using histogram-based dominant color estimation for realistic occlusions
- **Targets specific regions** within bounding boxes to simulate leaf occlusions and wheat head overlaps

### The Problem

Field conditions present significant challenges for wheat head detection:
- 🍃 Occlusions from leaves
- 🌾 Overlapping wheat heads
- ☀️ Varying lighting conditions
- 🏃 Motion blur

Traditional augmentation methods apply random masking without considering the spatial distribution of objects, often degrading performance when occlusions already exist.

## ✨ Key Features

- **Smart Mask Placement**: Uses IoU-based filtering to identify suitable candidates for masking
- **Adaptive Color Selection**: Histogram analysis determines the most realistic mask color
- **Probabilistic Sampling**: Controlled augmentation with fine-grained parameters
- **Architecture Agnostic**: Works across different detector types (Faster R-CNN, FCOS, DETR)
- **Easy Integration**: Compatible with popular data augmentation libraries

## 📊 Performance

BBoxCut achieves significant mAP improvements on the GWHD 2021 dataset:

| Model | Baseline | CutOut | Region-Aware RE | **BBoxCut (Ours)** | **Improvement** |
|-------|----------|--------|-----------------|-------------------|-----------------|
| Faster R-CNN | 51.14 | 51.29 | 52.23 | **53.90** | **+2.76** |
| FCOS | 57.18 | 57.87 | 58.82 | **60.44** | **+3.26** |
| DETR | 57.40 | 58.20 | 58.50 | **59.30** | **+1.90** |

### Qualitative Results

<div align="center">
  <img src="assets/qualitative_results.png" alt="Qualitative Comparison" width="800"/>
  <p><i>Left to right: Baseline, CutOut, Region-Aware RE, BBoxCut (Ours)</i></p>
</div>

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/BBoxCut.git
cd BBoxCut

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning==1.9.0
albumentations==1.0.0
pandas==1.4.1
numpy>=1.21.0
opencv-python-headless==4.5.5.64
scikit-learn==1.2.2
scikit-image==0.18.2
matplotlib==3.4.2
Pillow>=10.0.0
tqdm==4.61.1
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from wheat_dataset import WheatDataset
from torch.utils.data import DataLoader
from utils import collate_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define your data augmentation pipeline
train_transform = A.Compose([
    A.RandomResizedCrop(height=800, width=800),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Create dataset
dataset = WheatDataset(
    csv_file='path/to/train.csv',
    root_dir='path/to/images',
    transform=train_transform,
    image_set='train'
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)
```

### Training a Model

```python
from fastercnn import DAFasterRCNN
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Initialize model
model = DAFasterRCNN(
    n_classes=2,
    batchsize=2,
    n_vdomains=8,
    training_dataset=train_dataset
)

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='best_model',
    save_top_k=1,
    mode='min'
)

early_stop = EarlyStopping(
    monitor='val_acc',
    patience=10,
    mode='max'
)

# Initialize trainer
trainer = Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0,
    callbacks=[checkpoint_callback, early_stop]
)

# Train
trainer.fit(model, train_dataloader, val_dataloader)
```

## 🎨 Usage with Albumentations

BBoxCut can be easily integrated into Albumentations pipelines. Here's how to implement it as a custom transform:

```python
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import numpy as np
import cv2

class BBoxCut(DualTransform):
    """
    Apply BBoxCut augmentation: mask random regions within bounding boxes
    using dominant color estimation.
    
    Args:
        p_aug (float): Probability of applying the augmentation
        p_mask (float): Probability of masking each non-overlapping box
        alpha_w (float): Maximum width percentage to mask
        alpha_h (float): Maximum height percentage to mask
        iou_threshold (float): IoU threshold for detecting overlaps
        always_apply (bool): Whether to always apply the transform
        p (float): Probability of applying the transform
    """
    
    def __init__(
        self,
        p_aug=0.3,
        p_mask=0.3,
        alpha_w=0.3,
        alpha_h=0.3,
        iou_threshold=0.5,
        always_apply=False,
        p=0.5
    ):
        super().__init__(always_apply, p)
        self.p_aug = p_aug
        self.p_mask = p_mask
        self.alpha_w = alpha_w
        self.alpha_h = alpha_h
        self.iou_threshold = iou_threshold
    
    def get_dominant_color(self, image):
        """Extract dominant color using histogram analysis"""
        colors = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            colors.append(np.argmax(hist))
        return tuple(colors)
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def apply(self, img, bboxes=None, **params):
        """Apply BBoxCut augmentation"""
        if bboxes is None or len(bboxes) == 0:
            return img
        
        if np.random.random() > self.p_aug:
            return img
        
        img = img.copy()
        h, w = img.shape[:2]
        
        # Get dominant color
        dominant_color = self.get_dominant_color(img)
        
        # Filter overlapping boxes
        non_overlapping = []
        for i, box in enumerate(bboxes):
            is_overlapping = False
            for j, other_box in enumerate(bboxes):
                if i != j and self.compute_iou(box, other_box) > self.iou_threshold:
                    is_overlapping = True
                    break
            if not is_overlapping:
                non_overlapping.append(box)
        
        # Apply masking to selected boxes
        for box in non_overlapping:
            if np.random.random() < self.p_mask:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                box_w, box_h = x2 - x1, y2 - y1
                
                # Random mask dimensions
                mask_w = int(np.random.uniform(0, self.alpha_w * box_w))
                mask_h = int(np.random.uniform(0, self.alpha_h * box_h))
                
                # Random mask position within box
                mask_x = np.random.randint(x1, max(x1 + 1, x2 - mask_w))
                mask_y = np.random.randint(y1, max(y1 + 1, y2 - mask_h))
                
                # Apply mask
                img[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = dominant_color
        
        return img
    
    def get_transform_init_args_names(self):
        return ("p_aug", "p_mask", "alpha_w", "alpha_h", "iou_threshold")

# Usage in augmentation pipeline
transform = A.Compose([
    A.RandomResizedCrop(height=800, width=800),
    A.HorizontalFlip(p=0.5),
    BBoxCut(p_aug=0.3, p_mask=0.3, alpha_w=0.3, alpha_h=0.3, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Apply to image and bboxes
augmented = transform(image=image, bboxes=bboxes, labels=labels)
```

## 📁 Project Structure

```
BBoxCut/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── fastercnn.ipynb          # Jupyter notebook with training code
├── wheat_dataset.py          # Dataset class for GWHD 2021
├── utils.py                  # Utility functions (collate_fn, etc.)
│
├── assets/                   # Images for README
│   ├── bboxcut_demo.png
│   └── qualitative_results.png
│
├── configs/                  # Configuration files
│   └── train_config.yaml
│
├── checkpoints/              # Saved model checkpoints
│   └── .gitkeep
│
├── logs/                     # Training logs
│   └── .gitkeep
│
└── docs/                     # Additional documentation
    └── paper.pdf             # Research paper
```

## 📦 Dataset

This project uses the **Global Wheat Head Detection (GWHD) 2021** dataset.

### Download

1. Visit the [GWHD 2021 competition page](https://www.aicrowd.com/challenges/global-wheat-challenge-2021)
2. Download the dataset
3. Extract to your preferred location

### Dataset Structure

```
gwhd_2021/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── competition_train.csv
├── competition_val.csv
└── competition_test.csv
```

### CSV Format

The CSV files contain:
- `image_name`: Image filename
- `BoxesString`: Bounding boxes in format "x y w h; x y w h; ..."
- `domain`: Domain/location identifier

## 🏋️ Training

### Configuration

Modify training parameters in your training script or config file:

```python
# Training hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
MAX_EPOCHS = 50
WEIGHT_DECAY = 0.0001

# BBoxCut parameters
P_AUG = 0.3          # Probability of applying augmentation
P_MASK = 0.3         # Probability of masking each box
ALPHA_W = 0.3        # Max width percentage to mask
ALPHA_H = 0.3        # Max height percentage to mask
IOU_THRESHOLD = 0.5  # IoU threshold for overlap detection
```

### Run Training

```bash
# Single GPU
python train.py --config configs/train_config.yaml

# Or use the Jupyter notebook
jupyter notebook fastercnn.ipynb
```

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir logs/
```

## 📈 Results

### Quantitative Results

Improvements over baseline on GWHD 2021 test set:

- **Faster R-CNN**: 51.14 → 53.90 mAP (+2.76)
- **FCOS**: 57.18 → 60.44 mAP (+3.26)  
- **DETR**: 57.40 → 59.30 mAP (+1.90)

### Ablation Studies

Impact of mask color choice (Faster R-CNN):

| Mask Color | mAP |
|------------|-----|
| Black | 48.75 |
| Gray | 50.51 |
| White | 52.75 |
| Random | 52.70 |
| **Dominant (Ours)** | **53.90** |

## 📄 Citation

If you use BBoxCut in your research, please cite:

```bibtex
@article{gowri2025bboxcut,
  title={BBoxCut: A Targeted Data Augmentation Technique for Enhancing Wheat Head Detection Under Occlusions},
  author={Gowri P, Yasashwini Sai and Seemakurthy, Karthik and Opoku, Andrews Agyemang and Bharatula, Sita Devi},
  journal={arXiv preprint arXiv:2503.24032},
  year={2025}
}
```

## 🙏 Acknowledgments

- Research England (Lincoln Agri-Robotics) for funding support
- School of Computer Science, University of Lincoln, UK
- The GWHD 2021 dataset creators and contributors
- PyTorch and Albumentations communities

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

- **Yasashwini Sai Gowri P** - [GitHub](https://github.com/yourusername)
- **Project Link**: [https://github.com/yourusername/BBoxCut](https://github.com/yourusername/BBoxCut)

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐!

---

<div align="center">
  Made with ❤️ for advancing agricultural AI
</div>