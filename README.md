# 🧠 Brain Tumor Detection using MRI Images

A deep learning project for detecting and classifying brain tumors from MRI scans using a **Convolutional Neural Network (CNN)** based on **VGG16 architecture**.

---

## 📌 Project Overview
This project implements a **transfer learning approach** using **VGG16** as a base model to classify brain MRI images into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

✅ The model achieves **92% accuracy** on the test set with excellent **ROC–AUC scores** across all classes.

---

## 📂 Dataset
The dataset is sourced from Kaggle:  
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (CC0-1.0 License)

**Dataset size:**
- Training images: **2,852**
- Testing images: **1,311**
- Balanced distribution across four classes

---

## 🏗️ Model Architecture
The model uses **transfer learning** with the following architecture:

- **Base model:** VGG16 (pretrained on ImageNet, layers frozen)
- **Custom top layers:**
  - Flatten layer  
  - Dropout (0.3)  
  - Dense layer (128 units, ReLU activation)  
  - Dropout (0.2)  
  - Output layer (4 units, Softmax activation)  

---

## 📊 Training Results

### Accuracy and Loss Curves
<img width="792" height="393" alt="training" src="https://github.com/user-attachments/assets/5a5b5c06-74ec-4b8b-8459-083619bdd1d1" />

### Performance Metrics
```
              precision    recall  f1-score   support

           0       0.76      0.98      0.86       306
           1       0.99      0.98      0.99       300
           2       0.99      0.99      0.99       405
           3       1.00      0.72      0.84       300

    accuracy                           0.92      1311
   macro avg       0.94      0.92      0.92      1311
weighted avg       0.94      0.92      0.92      1311
```

### ROC–AUC Curves
<img width="846" height="701" alt="roc-auc" src="https://github.com/user-attachments/assets/0ee862b1-2ede-4260-9051-30132fea02b7" />


**AUC Scores:**
- Class 0 (Glioma): **0.99**
- Class 1 (Meningioma): **1.00**
- Class 2 (No Tumor): **1.00**
- Class 3 (Pituitary): **0.99**

---

## 🔍 Prediction Example
<img width="420" height="411" alt="brain" src="https://github.com/user-attachments/assets/42566d77-5d94-465c-ba76-7eef64d550f1" />

**Output:**  
`Tumor: Pituitary (Confidence: 99.97%)`

---

## ⚡ Installation and Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset from Kaggle**  
   Place the dataset into the project folder.

4. **Run the notebook**
   ```bash
   jupyter notebook "Brain Tumor Detection.ipynb"
   ```

---

## 📦 Dependencies
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- PIL / Pillow  

---

## 📁 File Structure
```
brain-tumor-detection/
├── Brain Tumor Detection.ipynb  # Main notebook
├── model.h5                     # Trained model
├── requirements.txt             # Dependencies
├── images/                      # Directory for result images
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── prediction_example.png
└── README.md
```

---

## 🚀 Future Improvements
- Experiment with advanced **data augmentation techniques**  
- Try different architectures (ResNet, EfficientNet)  
- Implement **class weighting** for imbalanced data  
- Develop a **web interface (Streamlit/Flask)** for easy predictions  

---

## 🏅 Performance Summary
- **Accuracy:** ~92%  
- **F1-score:** High across most classes  
- **ROC–AUC:** >0.95 for all tumor categories  

---

## 📜 License
This project is licensed under the MIT License.  

---
