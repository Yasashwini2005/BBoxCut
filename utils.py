import torch
import torch.nn.functional as F

def collate_fn(batch):
    # Extract images, targets, and domain labels from the batch
    images = [i['image'] for i in batch]  # Assuming batch is a list of dicts
    targets = [i['target'] for i in batch]
    domain_labels = [i.get('domain_label', None) for i in batch]  # Use .get to handle missing keys

    # Determine the maximum height and width for padding
    max_height = max(img.size(1) for img in images)
    max_width = max(img.size(2) for img in images)

    def pad_image(img):
        _, h, w = img.size()
        pad_bottom = max_height - h
        pad_right = max_width - w
        return F.pad(img, pad=(0, pad_right, 0, pad_bottom), mode='constant', value=0)

    # Pad images to the maximum size and stack them into a single tensor
    padded_images = [pad_image(img) for img in images]
    images_tensor = torch.stack(padded_images, dim=0)

    return images_tensor, targets, domain_labels
