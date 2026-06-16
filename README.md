
# Deep Learning Feature Learning for Breast Cancer Detection

A deep learning project that uses an **Autoencoder** for feature extraction and a **Neural Network Classifier** for breast cancer diagnosis prediction using the Wisconsin Breast Cancer Dataset.

---

## 📌 Overview

This project demonstrates how deep learning can be used for feature learning and binary classification. An autoencoder is first trained to learn compact feature representations from the dataset, and the encoded features are then used to train a classifier that predicts whether a tumor is **Malignant** or **Benign**.

---

## 📂 Dataset

**Dataset:** Wisconsin Breast Cancer Dataset

* Total Samples: **569**
* Features: **30**
* Target Classes:

  * **0 → Malignant**
  * **1 → Benign**

The dataset contains numerical features computed from digitized images of breast mass cell nuclei.

---

## ✨ Features

* Data preprocessing
* Feature scaling using StandardScaler
* Autoencoder-based feature extraction
* Neural Network classification
* Confusion matrix generation
* Model saving for future predictions

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Joblib

---

## 📁 Project Structure

```
deep-learning-feature-learning/
│
├── breast_cancer.csv           # Dataset
├── dl.py                       # Main program
├── encoder_model.keras         # Trained encoder model
├── classifier_model.keras      # Trained classifier model
├── scaler.pkl                  # Saved StandardScaler
├── confusion_matrix.png        # Evaluation result
└── README.md
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/pandu1437/deep-learning-feature-learning.git
```

Navigate to the project directory

```bash
cd deep-learning-feature-learning
```

Install the required libraries

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn joblib
```

---

## ▶ Running the Project

Execute the following command:

```bash
python dl.py
```

---

## 🔄 Project Workflow

```
Breast Cancer Dataset
          │
          ▼
Data Preprocessing
          │
          ▼
Feature Scaling
          │
          ▼
Train Autoencoder
          │
          ▼
Extract Encoded Features
          │
          ▼
Train Neural Network Classifier
          │
          ▼
Evaluate Performance
          │
          ▼
Save Models
```

---

## 🧠 Autoencoder

The autoencoder learns compressed feature representations from the original dataset.

### Training Configuration

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Epochs: 50

---

## 🤖 Classifier

The encoded features are used to train a feed-forward neural network for breast cancer prediction.

### Training Configuration

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Epochs: 40

---

## 📊 Results

**Model Accuracy:** **81.58%**

### Classification Performance

| Metric    | Malignant | Benign |
| --------- | --------: | -----: |
| Precision |      0.87 |   0.80 |
| Recall    |      0.60 |   0.94 |
| F1-Score  |      0.71 |   0.86 |

The project also generates a confusion matrix saved as:

```
confusion_matrix.png
```

---

## 💾 Output Files

After training, the following files are generated automatically:

* **encoder_model.keras** – Trained encoder model
* **classifier_model.keras** – Trained classifier
* **scaler.pkl** – Saved feature scaler
* **confusion_matrix.png** – Model evaluation visualization

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Deeper autoencoder architecture
* Early stopping
* Cross-validation
* Model deployment using Streamlit or Flask
* Explainable AI using SHAP or LIME

---

## 📚 Learning Outcomes

This project demonstrates:

* Data preprocessing
* Feature scaling
* Autoencoder-based feature learning
* Neural network classification
* Model evaluation
* TensorFlow/Keras implementation
* Saving trained models for deployment

---

## 👨‍💻 Author

**Pandu**

GitHub: https://github.com/pandu1437

---

## 📄 License

This project is developed for educational and learning purposes.

