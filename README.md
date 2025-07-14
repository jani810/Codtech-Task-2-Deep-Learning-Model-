# Codtech-Task-2-Deep-Learning-Model-
- COMPANY: CODTECH IT SOLUTIONS
- NAME: BHUKYA JANI
- INTERN ID: CT04DH2696
- DOMAIN: DATA SCIENCE
- DURATION: 4 WEEKS
- MENTOR: NEELA SANTOSH
---

## Overview
This project demonstrates a deep learning approach to classify flower species from the Iris dataset using TensorFlow and Keras. The Iris dataset contains 150 samples of iris flowers, with four features: sepal length, sepal width, petal length, and petal width. Each sample belongs to one of three classes: Setosa, Versicolor, or Virginica.

---

## Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  

---

## Workflow Summary

### 🔹 1. Data Preparation
- Loaded Iris dataset using Scikit-learn.
- Normalized feature values using `StandardScaler`.
- Encoded target labels (already numeric).
- Split the data into training and testing sets (80/20).

### 🔹 2. Model Architecture
- Input layer with 4 features
- Two Dense layers: 
  - Layer 1: 10 neurons, ReLU
  - Layer 2: 8 neurons, ReLU
- Output layer: 3 neurons (Softmax for multiclass classification)

### 🔹 3. Training & Evaluation
- Model trained for 50 epochs
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Achieved accuracy above 95% on the test set

### 🔹 4. Visualization
- Plotted training vs. validation accuracy and loss
- Helped monitor model performance across epochs

---

## Results
- **Final Test Accuracy:** ~96%  
- Model successfully distinguishes between the three Iris species  
- Training and validation curves show good generalization

---

## Files Included
- `iris_deep_learning.ipynb` – Full code for preprocessing, model training, and evaluation  
- `README.md` – Project documentation  
- `iris_accuracy_plot.png` (optional) – Accuracy/Loss visualization

---

## Conclusion
This project showcases a simple yet effective application of neural networks in classifying real-world structured data. It strengthens understanding of model building, training, evaluation, and visualization in a Data Science workflow.

