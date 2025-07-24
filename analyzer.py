import torch
import cv2
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import easyocr
from PIL import Image
import numpy as np
import io

# --- Define Device ---
# Automatically select GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Forgery Detection Model ---
class ForgeryDetector(nn.Module):
    def __init__(self):
        super(ForgeryDetector, self).__init__()
        self.model = resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

# --- Global Variables ---
FORGERY_MODEL_PATH = "models/forgery_detector.pth"
forgery_model = ForgeryDetector()

# Load model weights into the .model attribute
forgery_model.model.load_state_dict(torch.load(FORGERY_MODEL_PATH, map_location=device))

# Move model to device and set to evaluation mode
forgery_model.to(device)
forgery_model.eval()

# Initialize OCR reader
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Define image transformations for the forgery model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Main Analysis Function ---
def analyze_document(image_bytes: bytes) -> dict:
    """
    Analyzes a document image for forgery and performs OCR with preprocessing.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1. Forgery Detection
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = forgery_model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        # Assumes class 1 is 'tampered' based on alphabetical folder order
        tampered_prob = probabilities[1].item()

    forgery_result = {
        "is_tampered": bool(tampered_prob > 0.5),
        "tampered_confidence_score": f"{tampered_prob:.4f}"
    }

    # 2. Text Recognition (OCR) with Preprocessing
    image_np = np.array(image)
    
    # Convert image to grayscale for better processing
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to get a clear black and white image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Use the preprocessed image ('thresh') for OCR
    ocr_raw_results = ocr_reader.readtext(thresh)
    
    extracted_text = []
    for (bbox, text, prob) in ocr_raw_results:
        top_left = [int(point) for point in bbox[0]]
        bottom_right = [int(point) for point in bbox[2]]
        
        extracted_text.append({
            "text": text,
            "confidence": f"{prob:.4f}",
            "bounding_box": [top_left, bottom_right]
        })
    
    # 3. Combine Results
    final_result = {
        "forgery_analysis": forgery_result,
        "ocr_results": extracted_text
    }
    
    return final_result