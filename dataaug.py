import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from dataload import DataHandler

class PlantDataset(Dataset):
    def __init__(self, df, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "label"]
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, self.label_map[label]

class Augmentations:
    train = A.Compose([
        A.Resize(224, 224),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_dataloaders(batch_size=32):
    """Create DataLoaders with proper class mapping."""
    train_df, val_df, metadata_df = DataHandler.load_metadata()
    
    labels = sorted(metadata_df["label"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    train_dataset = PlantDataset(train_df, label_map, Augmentations.train)
    val_dataset = PlantDataset(val_df, label_map, Augmentations.val)
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        len(labels)
    )

if __name__ == "__main__":
    train_loader, val_loader, num_classes = get_dataloaders()
    print(f"Train Loader length: {len(train_loader)}")
    print(f"Validation Loader length: {len(val_loader)}")
    print(f"Number of classes: {num_classes}")
