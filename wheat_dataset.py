import pandas as pd
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import GeneralizedRCNNTransform

class WheatDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, image_set='train'):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_set = image_set

        # Initialize the RCNN Transform with default values
        self.rcnn_transform = GeneralizedRCNNTransform(
            min_size=800, 
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        image_np = np.array(image)  # Convert to numpy array

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        boxes_string = self.annotations.iloc[idx, 1]
        boxes = self.parse_boxes(boxes_string)
        print("Bounding boxes before tensor conversion:", boxes)

        boxes = np.array(boxes).astype('float').reshape(-1, 4).tolist()
        labels = [0] * len(boxes)  # Dummy labels

        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}

        image, target = self.rcnn_transform([image_tensor], [target])
        image = image[0].clone().detach()
        target = target[0]

        assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
        assert len(image.shape) == 3, f"Expected image of shape [3, H, W], got {image.shape}"
    # Assuming the domain is included in your CSV or dataset
        domain = self.annotations.iloc[idx, 2]  # Adjust column index as necessary
        return image, target, domain

    def __len__(self):
        return len(self.annotations)

    def parse_boxes(self, boxes_string):
        boxes = [list(map(float, box.split())) for box in boxes_string.split(';')]
        return boxes
