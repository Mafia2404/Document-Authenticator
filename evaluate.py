import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Configuration ---
MODEL_PATH = "models/forgery_detector.pth"
TEST_DATA_DIR = "data/test"
BATCH_SIZE = 32

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
except ImportError:
    print("Please install scikit-learn and seaborn: pip install scikit-learn seaborn pandas")
    exit()

# --- Model Definition ---
class ForgeryDetector(nn.Module):
    def __init__(self):
        super(ForgeryDetector, self).__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

# --- Main Evaluation Logic ---
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations for the test set
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the test dataset
    image_dataset = datasets.ImageFolder(TEST_DATA_DIR, data_transform)
    dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = image_dataset.classes
    print(f"Classes found: {class_names}")


    # Initialize model and load the saved weights
    model = ForgeryDetector().to(device)
    # --- THIS IS THE CORRECTED LINE ---
    model.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode

    all_preds = []
    all_labels = []

    print("Running evaluation on the test set...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print Performance Metrics
    print("\n--- Evaluation Report ---")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Display Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()