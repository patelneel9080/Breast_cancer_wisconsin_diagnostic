# Breast Cancer Wisconsin Diagnostic — Deep Learning PR 1

## 📌 Project Overview

This project focuses on building and evaluating deep learning models for breast cancer classification using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

The project explores Single-Layer Perceptrons (SLP), Multi-Layer Perceptrons (MLP), activation functions, feature scaling, Early Stopping, Dropout, and regularization techniques to improve model generalization.

The implementation uses Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn.

## 🎯 Objectives

- Perform exploratory data analysis (EDA).
- Preprocess and standardize numerical features.
- Build a Single-Layer Perceptron baseline model.
- Compare ReLU, Tanh, and Sigmoid activation functions.
- Understand and implement Early Stopping.
- Compare different Dropout rates.
- Apply L1, L2, and L1-L2 regularization.
- Build a final combined deep learning model.
- Evaluate models using classification metrics.
- Interpret results for medical decision-support applications.

## 📂 Dataset Information

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** Scikit-learn / UCI Machine Learning Repository

**Original dataset reference:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

| Property | Description |
|---|---|
| Total samples | 569 |
| Input features | 30 numerical features |
| Total CSV columns | 31 |
| Target column | `Diagnosis` |
| Malignant cases | 212 |
| Benign cases | 357 |
| Missing values | 0 |
| Duplicate rows | 0 |

### Target Encoding

The provided CSV uses the following target labels:

- `M` → `0` — Malignant
- `B` → `1` — Benign

The target encoding follows the assignment's specified convention.

## 🛠️ Technologies Used

- Python
- TensorFlow 2.x
- Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

## 📚 Project Tasks

### Task 1 — Data Loading, EDA & Preprocessing

- Load and inspect the dataset.
- Check dataset shape, data types, missing values, and duplicates.
- Visualize the target class distribution.
- Generate a feature correlation heatmap.
- Split the data into training and testing sets using stratification.
- Apply StandardScaler using training data only.

**Configuration:** `test_size=0.2`, `random_state=42`, `stratify=y`.

### Task 2 — Single-Layer Perceptron (SLP)

- Build a single sigmoid neuron with 30 input features.
- Compile using the Adam optimizer and binary cross-entropy.
- Train for 50 epochs.
- Plot training and validation loss and accuracy.
- Evaluate the model using test accuracy, classification report, and confusion matrix.

**Architecture:** `30 → 1`

**Trainable parameters:** 31.

### Task 3 — Multi-Layer Perceptron (MLP) and Activation Functions

- Build the MLP using ReLU hidden layers.
- Compare ReLU, Tanh, and Sigmoid hidden-layer activations.
- Plot training and validation accuracy.
- Select the best activation variant based on validation performance.
- Evaluate the selected model on the test dataset.

**Architecture:** `30 → 64 → 32 → 1`

### Task 4 — Early Stopping

- Build an MLP with 128 and 64 hidden neurons.
- Train with Early Stopping using validation loss.
- Restore the best model weights.
- Plot the stopping epoch and validation-loss curves.
- Compare Early Stopping against training without a callback.

**Configuration:** `patience=15`, `restore_best_weights=True`.

### Task 5 — Dropout Layers

- Implement Dropout regularization.
- Compare dropout rates of 0.1, 0.3, and 0.5.
- Use Early Stopping during training.
- Plot validation accuracy for the different dropout configurations.
- Evaluate the best-performing dropout model.

### Task 6 — Regularization

Implement and compare three regularization techniques:

- **L1 Regularization:** encourages sparse weights.
- **L2 Regularization:** penalizes large weights.
- **L1-L2 Regularization:** combines sparsity and weight shrinkage.

Compare training and validation loss and evaluate test accuracy.

### Task 7 — Final Model and Clinical Insights

Build a combined model incorporating:

- Dense layers with ReLU activation.
- L2 regularization.
- Dropout with a rate of 0.3.
- Early Stopping with restored best weights.

Compare the required models using:

- Test Accuracy
- Test Precision
- Test Recall
- Test F1-Score

Include clinical interpretation focusing on the importance of identifying malignant cases and reducing false negatives.

## 🔄 Methodology

The project follows this workflow:

1. Dataset loading and validation.
2. Exploratory data analysis.
3. Target encoding.
4. Stratified train-test split.
5. Feature scaling.
6. Baseline SLP training.
7. MLP and activation-function comparison.
8. Early Stopping experiment.
9. Dropout-rate comparison.
10. L1, L2, and L1-L2 regularization.
11. Final combined model.
12. Model evaluation and clinical interpretation.

## 📊 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness |
| Precision | Measures the proportion of predicted positives that are correct |
| Recall | Measures the proportion of actual positives correctly identified |
| F1-Score | Balances precision and recall |
| Confusion Matrix | Displays correct and incorrect classifications |

For medical decision support, recall for malignant cases is especially important because false negatives can result in missed malignant diagnoses.

## 📈 Visualizations

The notebook generates the following visualizations:

- Target class distribution
- Feature correlation heatmap
- SLP training and validation curves
- Activation-function comparison
- SLP and MLP confusion matrices
- Early Stopping training curves
- Early Stopping versus no-callback comparison
- Dropout-rate comparison
- Regularization loss comparison
- Test accuracy comparison
- Final model confusion matrix

All plots include descriptive titles and labeled axes.

## 📁 Repository Structure

```text
Breast_cancer_wisconsin_diagnostic/
│
├── breast_cancer_wisconsin_diagnostic.csv
├── breast_cancer_wisconsin_diagnostic.ipynb
├── breast_cancer_wisconsin_diagnostic.html
├── README.md
├── requirements.txt
```

*Note: Add the HTML file, plots, and saved model after generating them. The notebook currently displays its plots inline; save individual figures if you want to include them in the `plots/` folder.*

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/patelneel9080/Breast_cancer_wisconsin_diagnostic.git
```

### 2. Open the project directory

```bash
cd Breast_cancer_wisconsin_diagnostic
```

### 3. Run the notebook

Open the notebook using Jupyter Notebook or Google Colab.

For a local Jupyter installation:

```bash
jupyter notebook
```

Ensure the CSV file is available in the notebook's working directory before running the cells.

## 📦 Requirements

The project dependencies are listed in `requirements.txt`:

```text
tensorflow>=2.12.0,<3.0.0
scikit-learn>=1.4.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<3.0.0
matplotlib>=3.7.0,<4.0.0
seaborn>=0.13.0,<1.0.0
jupyter>=1.0.0,<2.0.0
```

## 🏥 Clinical Insight

The project demonstrates how neural networks can classify breast cancer cases using numerical cell-nucleus measurements.

In a clinical decision-support setting, selecting a model based only on accuracy may be insufficient. Missing a malignant case is generally more costly than producing a false positive, so malignant-case recall and the precision-recall trade-off should be carefully evaluated.

A classification threshold below 0.5 may be investigated to improve malignant-case detection, but it should be selected using appropriate clinical validation.

**Important:** This project is for educational purposes and is not a substitute for professional medical diagnosis or clinical validation.

## 🚀 Future Improvements

- Perform hyperparameter tuning.
- Compare additional neural-network architectures.
- Optimize the classification threshold using validation data.
- Evaluate recall and precision specifically for malignant cases.
- Add cross-validation and calibration analysis.
- Package the trained model for a validated prediction interface.

## 👨‍💻 Author

**Neel Patel**

GitHub: https://github.com/patelneel9080

## 📄 License

This project is developed for educational purposes as part of the Deep Learning PR 1 assignment. Refer to the dataset source for the applicable dataset terms.
