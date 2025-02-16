import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report
from dataaug import get_dataloaders
from model import PlantClassifier

if __name__ == "__main__":
    try:
        # Load data
        _, val_loader, num_classes = get_dataloaders()
        
        # Load metadata for target names
        df = pd.read_csv("metadata.csv")
        target_names = sorted(df["label"].unique())
        
        # Initialize model with correct number of classes (should be 70)
        classifier = PlantClassifier(num_classes)
        
        # Load checkpoint from training (ensure it's trained on 70 classes)
        checkpoint_path = "best_model.pth"
        classifier.model.load_state_dict(torch.load(checkpoint_path, map_location=classifier.device))
        classifier.model.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(classifier.device), labels.to(classifier.device)
                outputs = classifier.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print(classification_report(all_labels, all_preds, target_names=target_names))
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
