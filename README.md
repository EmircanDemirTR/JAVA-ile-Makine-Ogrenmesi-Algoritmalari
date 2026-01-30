# Machine Learning Algorithms in Java

A comprehensive collection of Machine Learning algorithms implemented in Java using **SMILE** and **Weka** libraries. This project covers the full ML pipeline: data preprocessing, model training, evaluation, and ensemble methods.

## Implemented Algorithms

### Data Preprocessing (bolum04)
- File & internet data loading
- Missing value imputation
- Outlier detection
- Categorical encoding
- Feature scaling / normalization
- Class imbalance handling (SMOTE)

### Model Training (bolum05)
- Train/test split strategies
- Model training pipelines

### Regression (bolum06)
- Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Decision Tree Regression

### Regression Evaluation (bolum07)
- Error metrics (MAE, MSE, RMSE, R²)
- Residual analysis
- Metric interpretation

### Classification (bolum08)
- K-Nearest Neighbors (KNN) with weighted voting & distance metrics
- Decision Tree with pruning & visualization
- Logistic Regression with probability analysis
- Naive Bayes with model comparison
- Support Vector Machine (SVM) with kernel comparison

### Classification Evaluation (bolum09)
- Confusion Matrix analysis
- ROC / AUC curves
- Precision, Recall, F1-Score

### Ensemble Methods (bolum10)
- Random Forest (classification & regression)
- Gradient Boosting
- AdaBoost

## Tech Stack

- **Language:** Java 25
- **Build Tool:** Maven
- **ML Libraries:** SMILE 5.1.0, Weka 3.8.6
- **Logging:** SLF4J

## Getting Started

### Prerequisites
- Java 25+
- Maven 3.8+

### Run

```bash
git clone https://github.com/EmircanDemirTR/JAVA-ile-Makine-Ogrenmesi-Algoritmalari.git
cd JAVA-ile-Makine-Ogrenmesi-Algoritmalari
mvn compile
mvn exec:java -Dexec.mainClass="com.btkakademi.ml.bolum08.SmileKNN"
```

Replace the main class path with any algorithm you want to run.

## Datasets

The `src/main/resources/datasets/` directory includes commonly used ML datasets:
- Iris, Wine, Glass, Breast Cancer, Mushroom
- Boston Housing, Auto MPG
- Student Performance, Wine Quality

## License

MIT
