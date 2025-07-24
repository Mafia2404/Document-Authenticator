import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler # <-- 1. IMPORT SCHEDULER
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import time

# --- Configuration ---
DATA_DIR = "data"
MODEL_SAVE_PATH = "models/forgery_detector.pth"
NUM_EPOCHS = 25 # <-- Increased epochs for more training time
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- Data Transformation and Loading ---
# Added more augmentations like ColorJitter and RandomRotation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #<-- New
        transforms.RandomRotation(10), #<-- New
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Classes: {class_names}")

# --- Model Setup ---
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

# We no longer freeze the layers, so we remove the loop that sets requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# --- 2. IMPROVEMENT: Use AdamW and train all parameters ---
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 3. IMPROVEMENT: Add a Learning Rate Scheduler ---
# Decays the learning rate by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# --- Training Loop ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25): #<-- Pass scheduler
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}'); print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step() #<-- Update learning rate

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Save the best model weights
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New best model saved with accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    return model


if __name__ == '__main__':
    if not os.path.exists("models"): os.makedirs("models")

    # Pass the scheduler to the training function
    train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
    print("Training finished. Best model weights are saved in models/forgery_detector.pth")