# Document-Authenticator
# End-to-End Document Authenticator

![Live Demo GIF](link_to_your_gif_here.gif)

This project is a full-stack deep learning application that analyzes document images to detect digital tampering and extracts text content using Optical Character Recognition (OCR). The entire machine learning lifecycle is demonstrated, from data preparation and model training to final deployment via a REST API with a web-based user interface.

---

## 🚀 Key Features
- **Deep Learning Forgery Detection:** A fine-tuned ResNet18 model classifies documents as either `authentic` or `tampered`.
- **Text Recognition (OCR):** Extracts all machine-readable text from the document image using an OCR engine.
- **Image Preprocessing:** Utilizes OpenCV to automatically preprocess images (grayscale, thresholding) to improve OCR accuracy.
- **REST API:** The backend is built with FastAPI, exposing a simple `/analyze` endpoint.
- **Interactive UI:** A user-friendly frontend built with HTML, CSS, and vanilla JavaScript allows for easy image uploads and result visualization.

---

## 📈 Performance
The forgery detection model was evaluated on a held-out test set, achieving **95% overall accuracy**.

#### Classification Report
```
              precision    recall  f1-score   support

   authentic       0.93      0.97      0.95        40
    tampered       0.98      0.93      0.95        45

    accuracy                           0.95        85
   macro avg       0.95      0.95      0.95        85
weighted avg       0.95      0.95      0.95        85
```
*(You should also add the Confusion Matrix image here)*

---

## 🛠️ Technologies Used
- **Backend:** Python, PyTorch, FastAPI, OpenCV, EasyOCR
- **Frontend:** HTML, CSS, JavaScript
- **Data & ML:** Scikit-learn, Pandas, Matplotlib, Seaborn
- **Tools:** Git, GitHub, Visual Studio Code

---

## ⚙️ How to Run
1.  **Clone the repository:** `gh repo clone Mafia2404/Document-Authenticator`
2.  **Create and activate a virtual environment:** `python -m venv venv` & `source venv/bin/activate`
3.  **Install dependencies:** `pip install -r requirements.txt`
4.  **Prepare Data (Optional):** Run `python prepare_dataset.py` to organize a local dataset.
5.  **Train Model (Optional):** Run `python train_forgery.py` to train the model. A pre-trained model is included.
6.  **Run the Backend Server:** `uvicorn app:app --reload`
7.  **Open the UI:** Open the `index.html` file in your browser.
