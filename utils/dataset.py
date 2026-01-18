import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import random


def get_train_transforms():
    """Get augmentation transforms for training"""
    return T.Compose([
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        T.RandomAutocontrast(p=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms():
    """Get transforms for validation (no augmentation)"""
    return None


class TrafficSignDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None, augment=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.augment = augment
        
        # Set up augmentation
        if augment:
            self.aug_transforms = get_train_transforms()
        else:
            self.aug_transforms = None
        
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        print(f"Found {len(self.image_files)} images in {images_dir}")
        if augment:
            print("  -> Data augmentation ENABLED")
        
    def __len__(self):
        return len(self.image_files)
    
    def _apply_geometric_augmentation(self, image, boxes):
        """Apply geometric augmentations that affect both image and boxes"""
        img_width, img_height = image.size
        
        # Random horizontal flip (50% probability)
        if random.random() < 0.5:
            image = T.functional.hflip(image)
            if len(boxes) > 0:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                boxes_tensor[:, [0, 2]] = img_width - boxes_tensor[:, [2, 0]]
                boxes = boxes_tensor.tolist()
        
        # Random scale (resize between 0.8 and 1.2)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.8, 1.2)
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            image = image.resize((new_width, new_height), Image.BILINEAR)
            
            if len(boxes) > 0:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                boxes_tensor *= scale_factor
                # Clamp to new image boundaries
                boxes_tensor[:, [0, 2]] = boxes_tensor[:, [0, 2]].clamp(0, new_width)
                boxes_tensor[:, [1, 3]] = boxes_tensor[:, [1, 3]].clamp(0, new_height)
                boxes = boxes_tensor.tolist()
        
        return image, boxes
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        # Load corresponding label file
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        # Convert to x1, y1, x2, y2 format
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        # Clamp to image boundaries
                        x1 = max(0, min(x1, img_width))
                        y1 = max(0, min(y1, img_height))
                        x2 = max(0, min(x2, img_width))
                        y2 = max(0, min(y2, img_height))
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id + 1)  # +1 because 0 is background
        
        # Apply geometric augmentation (affects boxes too)
        if self.augment and len(boxes) > 0:
            image, boxes = self._apply_geometric_augmentation(image, boxes)
            img_width, img_height = image.size
        
        # Convert image to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        
        # Apply color augmentation (only affects image, not boxes)
        if self.augment and self.aug_transforms:
            image = self.aug_transforms(image)
        
        # Apply custom transforms if provided
        if self.transforms:
            image = self.transforms(image)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }
            
        return image, target