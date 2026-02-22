# 🧠 Olivetti Face Recognition using PCA + SVM

A Machine Learning project that performs face recognition using the Olivetti Faces dataset from Scikit-learn.

This project demonstrates:
- Face data visualization
- Dimensionality reduction using PCA
- Classification using Support Vector Machine (SVM)
- Model evaluation using standard ML metrics

---

## 📂 Dataset

The project uses the built-in dataset:

`sklearn.datasets.fetch_olivetti_faces()`

- 400 grayscale face images
- 40 individuals
- 10 images per person
- Image size: 64x64 pixels

---

## 🚀 Project Workflow

### 1️⃣ Data Loading
- Load Olivetti face dataset
- Extract features and labels

### 2️⃣ Data Visualization
- Display sample face images
- Understand data distribution

### 3️⃣ Preprocessing
- Train-Test Split
- Feature scaling using `StandardScaler`

### 4️⃣ Dimensionality Reduction
- Principal Component Analysis (PCA)
- Reduce high dimensional image data
- Speed up training and improve performance

### 5️⃣ Model Training
- Support Vector Classifier (SVC)
- Implemented using a Scikit-learn Pipeline:
    - StandardScaler
    - PCA
    - SVC

### 6️⃣ Model Evaluation
- Accuracy Score
- Classification Report
- Confusion Matrix
- Visualization using Seaborn

---

## 📊 Evaluation Metrics

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ Confusion Matrix Heatmap

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📈 Results

The model successfully classifies faces using PCA for dimensionality reduction and SVM for classification.

Dimensionality reduction significantly improves computational efficiency while maintaining strong classification accuracy.

---

## 💡 Key Learning Outcomes

- Handling image datasets in ML
- Feature scaling importance
- PCA for dimensionality reduction
- Building ML pipelines
- Evaluating classification models properly

---

## ▶️ How to Run

1. Clone the repository
cd olivetti-face-recognition-ml


2. Create virtual environment (optional but recommended)


python -m venv env
source env/bin/activate # Linux/Mac
env\Scripts\activate # Windows


3. Install dependencies


pip install -r requirements.txt


4. Run Jupyter Notebook


jupyter notebook

---

## 📌 Future Improvements

- Try MLPClassifier
- Hyperparameter tuning (GridSearchCV)
- Compare multiple models
- Deploy using Streamlit
- Face recognition with CNN (Deep Learning)

---

## 👤 Author

Devendra Kushwah

If you found this helpful, consider giving ⭐ to the repo!
