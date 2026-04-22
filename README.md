# 🐶🐱 Cats vs Dogs Classification using VGG16 + SVM
## 📌 Project Overview

This project focuses on classifying images of cats and dogs using a **hybrid machine learning approach**.
Instead of training a deep neural network from scratch, I used:
* **VGG16** for feature extraction
* **Support Vector Machine (SVM)** for final classification
This approach achieves **high accuracy (97%)** while keeping computation efficient.

## 🚀 Key Highlights
* ✅ Achieved **97% accuracy**
* ✅ Used **transfer learning** (pretrained VGG16)
* ✅ Skipped corrupt/broken images automatically
* ✅ Reduced feature size using Global Average Pooling
* ✅ Combined deep learning + classical ML (SVM)

## ⚙️ Tech Stack
* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* tqdm

## 🧠 How the Model Works
### 1. Data Loading
* Images are loaded from folders
* Corrupt images are skipped using error handling

### 2. Preprocessing
* Images resized to **224 × 224**
* Preprocessed using VGG16 format

### 3. Feature Extraction
* Used pretrained VGG16 (ImageNet weights)
* Removed top layers
* Applied Global Average Pooling
* Output: **512 features per image**

### 4. Model Training
* Features scaled using StandardScaler
* Data split into training (80%) and testing (20%)
* Trained SVM with RBF kernel

### 5. Evaluation
* Accuracy and classification report generated

## ⚡ Performance Insights
* Feature shape: **(4000, 512)**
* Training samples: 3200
* Testing samples: 800
* Fast training due to feature extraction approach

## 💡 Key Learnings
* Transfer learning significantly improves performance
* Hybrid models (CNN + SVM) are efficient and powerful
* Data cleaning (removing corrupt images) is essential
* Feature scaling improves SVM performance

## 🔮 Future Improvements
* Add data augmentation
* Try other classifiers (Random Forest, XGBoost)
* Use advanced models like ResNet
* Build a web app using Flask or Streamlit

## ⭐ Conclusion
This project demonstrates how combining deep learning with traditional machine learning can achieve **high accuracy with low computational cost**, making it ideal for real-world applications on limited hardware.
