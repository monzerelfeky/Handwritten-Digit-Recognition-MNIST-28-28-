# Handwritten Digit Recognition (MNIST 28×28)

This project implements a **handwritten digit recognition system** using the **MNIST 28×28 dataset**.
Multiple classical machine learning models are trained, evaluated, and compared using a unified preprocessing and feature-selection pipeline.

---

## Project Overview

* Task: Multiclass classification of handwritten digits (0–9)
* Dataset: MNIST (CSV format)
* Input: 28×28 grayscale images (flattened to 784 features)
* Output: Predicted digit label
* Goal: Compare classical ML models and select the best performer

---

## Project Structure

```
.
├── mnist_train.csv                   # Training dataset
├── mnist_test.csv                    # Test dataset
├── mnist_28x28.py                    # Main training & evaluation script
├── mnist_28x28.ipynb                 # Main training & evaluation notebook
├── mnist_28x28_model_results.csv     # Model accuracy comparison
├── mnist_28x28_confusion_matrix.png  # Confusion matrix of best model
└── README.md
```

---

## Dataset

* **Source:** MNIST handwritten digit dataset
* **Classes:** 10 (digits 0–9)
* **Training samples:** 60,000
* **Testing samples:** 10,000
* **Features:** 784 pixel values per image

Each CSV file contains:

* `label` column → digit class
* 784 pixel columns → grayscale intensities

---

## Preprocessing & Feature Selection

All models use the same pipeline to ensure fair comparison.

### Preprocessing

* **MinMaxScaler**

  * Scales pixel values to range [0, 1]
  * Required for KNN, SVM, and neural networks

### Feature Selection

* **SelectKBest (Chi-Square test)**
* Top **200 features** selected
* Reduces dimensionality and noise

---

## Models Trained

* Decision Tree
* Random Forest (200 estimators)
* K-Nearest Neighbors (k = 5)
* Support Vector Machine (RBF kernel)
* Artificial Neural Network (MLP)
* Gaussian Naive Bayes

All models are implemented using **scikit-learn Pipelines**.

---

## Evaluation

* **Metric:** Accuracy
* **Analysis tools:**

  * Confusion matrix
  * Sample prediction visualization

---

## Results

| Model         | Accuracy   |
| ------------- | ---------- |
| **SVM**       | **96.77%** |
| ANN (MLP)     | 96.16%     |
| Random Forest | 95.30%     |
| KNN           | 94.80%     |
| Decision Tree | 86.01%     |
| Naive Bayes   | 71.19%     |

### Best Model

**Support Vector Machine (SVM)**

* Handles high-dimensional data well
* RBF kernel captures non-linear digit patterns
* Strong generalization on test data

---

## How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 2. Place Dataset Files
install the datasets from here: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download

Ensure these files are in the project directory:

* `mnist_train.csv`
* `mnist_test.csv`

### 3. Run the Script

```bash
python mnist_28x28.py
```

### 4. Outputs

* Printed accuracy for each model
* CSV file with results
* Confusion matrix image
* Sample prediction plots

---

## Limitations

* Classical ML models only (no CNNs)
* Spatial pixel relationships are not preserved
* Limited hyperparameter tuning
* SVM training can be computationally expensive

---

## References

* MNIST Handwritten Digit Database
* scikit-learn Documentation
  [https://scikit-learn.org](https://scikit-learn.org)
* Chi-Square Feature Selection
  [https://scikit-learn.org/stable/modules/feature_selection.html](https://scikit-learn.org/stable/modules/feature_selection.html)
