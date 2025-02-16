import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from dataaug import get_dataloaders

class PlantClassifier:
    def __init__(self, num_classes):
        # Initialize pre-trained ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        print(f"Training on {self.device}")
        print(f"Number of classes: {num_classes}")

    def train(self, train_loader, val_loader, epochs=15):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
        
        best_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pth")
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        return history
    
    def evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                total_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / len(loader.dataset)
        return avg_loss, accuracy

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

if __name__ == "__main__":
    try:
        train_loader, val_loader, num_classes = get_dataloaders()
        classifier = PlantClassifier(num_classes)
        history = classifier.train(train_loader, val_loader)
        torch.save(classifier.model.state_dict(), "final_model.pth")
        plot_training(history)
    except Exception as e:
        print(f"Critical Error: {str(e)}")
