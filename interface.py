import argparse
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from dataaug import get_dataloaders, Augmentations
from model import PlantClassifier

def train_model(epochs):
    train_loader, val_loader, num_classes = get_dataloaders()
    classifier = PlantClassifier(num_classes)
    history = classifier.train(train_loader, val_loader, epochs=epochs)
    torch.save(classifier.model.state_dict(), "final_model.pth")
    print("Training complete. Final model saved as final_model.pth.")
    plot_training(history)

def evaluate_model(checkpoint_path):
    _, val_loader, num_classes = get_dataloaders()
    df = pd.read_csv("metadata.csv")
    target_names = sorted(df["label"].unique())
    classifier = PlantClassifier(num_classes)
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
    
    from sklearn.metrics import classification_report
    print("Evaluation Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

def predict_image(image_path, checkpoint_path):
    df = pd.read_csv("metadata.csv")
    label_list = sorted(df["label"].unique())
    num_classes = len(label_list)
    classifier = PlantClassifier(num_classes)
    classifier.model.load_state_dict(torch.load(checkpoint_path, map_location=classifier.device))
    classifier.model.eval()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = Augmentations.val(image=image)
    tensor_image = transformed["image"].unsqueeze(0).to(classifier.device)
    
    with torch.no_grad():
        outputs = classifier.model(tensor_image)
        _, pred_idx = torch.max(outputs, 1)
    
    predicted_label = label_list[pred_idx.item()]
    print(f"Predicted label for {image_path}: {predicted_label}")

def plot_training(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plant Identification Interface")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to the saved checkpoint")
    
    pred_parser = subparsers.add_parser("predict", help="Predict a single image")
    pred_parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    pred_parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to the saved checkpoint")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args.epochs)
    elif args.command == "evaluate":
        evaluate_model(args.checkpoint)
    elif args.command == "predict":
        predict_image(args.image, args.checkpoint)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
