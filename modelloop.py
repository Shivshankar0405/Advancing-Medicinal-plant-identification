#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torchvision import models
#import matplotlib.pyplot as plt
#from dataaug import get_dataloaders
#
#class PlantModel:
#    def __init__(self, num_classes):
#        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#        num_ftrs = self.model.fc.in_features
#        self.model.fc = nn.Linear(num_ftrs, num_classes)
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        self.model = self.model.to(self.device)
#        
#    def train_model(self, train_loader, val_loader, epochs=15):
#        criterion = nn.CrossEntropyLoss()
#        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
#        
#        best_acc = 0.0
#        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
#        
#        for epoch in range(epochs):
#            # Training phase
#            self.model.train()
#            running_loss = 0.0
#            for inputs, labels in train_loader:
#                inputs, labels = inputs.to(self.device), labels.to(self.device)
#                
#                optimizer.zero_grad()
#                outputs = self.model(inputs)
#                loss = criterion(outputs, labels)
#                loss.backward()
#                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
#                optimizer.step()
#                running_loss += loss.item()
#                
#            # Validation phase
#            val_loss, val_acc = self.evaluate(val_loader, criterion)
#            
#            # Update learning rate
#            scheduler.step(val_acc)
#            
#            # Store metrics
#            history['train_loss'].append(running_loss/len(train_loader))
#            history['val_loss'].append(val_loss)
#            history['val_acc'].append(val_acc)
#            
#            # Save best model
#            if val_acc > best_acc:
#                best_acc = val_acc
#                torch.save(self.model.state_dict(), "best_model.pth")
#                
#            # Print epoch summary
#            print(f"Epoch {epoch+1}/{epochs}")
#            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
#            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
#            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")
#            
#        return history
#    
#    def evaluate(self, loader, criterion):
#        self.model.eval()
#        running_loss = 0.0
#        correct = 0
#        total = 0
#        
#        with torch.no_grad():
#            for inputs, labels in loader:
#                inputs, labels = inputs.to(self.device), labels.to(self.device)
#                outputs = self.model(inputs)
#                loss = criterion(outputs, labels)
#                
#                running_loss += loss.item()
#                _, preds = torch.max(outputs, 1)
#                correct += (preds == labels).sum().item()
#                total += labels.size(0)
#                
#        return running_loss/len(loader), 100 * correct/total
#
#def plot_history(history):
#    plt.figure(figsize=(12, 5))
#    
#    plt.subplot(1, 2, 1)
#    plt.plot(history['train_loss'], label='Train Loss')
#    plt.plot(history['val_loss'], label='Val Loss')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.legend()
#    
#    plt.subplot(1, 2, 2)
#    plt.plot(history['val_acc'], label='Validation Accuracy')
#    plt.xlabel('Epochs')
#    plt.ylabel('Accuracy (%)')
#    plt.legend()
#    
#    plt.tight_layout()
#    plt.savefig("training_history.png")
#    plt.show()
#
#if __name__ == "__main__":
#    try:
#        # Initialize data
#        train_loader, val_loader, num_classes = get_dataloaders()
#        
#        # Initialize model
#        model = PlantModel(num_classes)
#        print(f"Training on {model.device} with {num_classes} classes")
#        
#        # Train
#        history = model.train_model(train_loader, val_loader)
#        
#        # Save and plot
#        torch.save(model.model.state_dict(), "final_model.pth")
#        plot_history(history)
#        
#    except Exception as e:
#        print(f"Error: {str(e)}")