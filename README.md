# Software Defect Prediction using Machine Learning

## a) Problem Statement

The objective of this project is to predict whether a software module will contain defects (`defect = 1`) or not (`defect = 0`) based on various software engineering metrics such as code complexity, coupling, test coverage, security vulnerabilities, and other quality indicators.

This is a supervised binary classification problem where the target variable is **`defect`**.

The goal is to compare multiple machine learning models and evaluate their performance using standard classification metrics.

---

## b) Dataset Description

The dataset consists of software engineering metrics extracted from software modules. Each row represents a module and includes the following types of features:

- Code complexity metrics (e.g., cyclomatic complexity)
- Size metrics (e.g., lines of code)
- Development activity metrics (e.g., commit frequency, bug fix commits)
- Code quality indicators (e.g., duplication percentage, coupling)
- Security and performance metrics
- Target variable: `defect` (0 = No defect, 1 = Defect)

The dataset is imbalanced, meaning one class appears more frequently than the other.

To address this imbalance:
- Logistic Regression, Decision Tree, and Random Forest were trained using `class_weight="balanced"` where applicable.
- Evaluation focused on F1 and MCC in addition to accuracy.

---

## c) Models Used

The following six models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was trained on the training split and evaluated on the test split using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model Name        | Accuracy  | AUC      | Precision | Recall   | F1       | MCC      |
|----------------------|----------:|---------:|----------:|---------:|---------:|---------:|
| Logistic Regression  | 0.932917  | 0.987953 | 0.999447  | 0.931387 | 0.964217 | 0.526787 |
| Decision Tree        | 1.000000  | 1.000000 | 1.000000  | 1.000000 | 1.000000 | 1.000000 |
| kNN                  | 0.971250  | 0.948236 | 0.971305  | 0.999914 | 0.985402 | 0.165645 |
| Naive Bayes          | 0.993000  | 0.998863 | 0.994018  | 0.999798 | 0.996402 | 0.871370 |
| Random Forest        | 0.999917  | 1.000000 | 0.999914  | 1.000000 | 0.999957 | 0.998548 |
| XGBoost              | 0.999833  | 1.000000 | 1.000000  | 0.999828 | 0.999914 | 0.997109 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performs reasonably well but struggles compared to ensemble methods. Lower MCC indicates weaker correlation despite high precision. |
| Decision Tree | Achieved perfect scores. This may indicate overfitting or highly separable data. |
| kNN | High recall but relatively low MCC. Sensitive to scaling and class imbalance. |
| Naive Bayes | Strong performance despite independence assumption. High recall and F1 score. |
| Random Forest (Ensemble) | Excellent overall performance. Very high F1 and MCC indicate strong generalization. |
| XGBoost (Ensemble) | Performs almost as well as Random Forest. High AUC and MCC show strong discriminative power. |

---

## Final Conclusion

Ensemble models (Random Forest and XGBoost) performed the best on this dataset, achieving near-perfect evaluation metrics.

However, extremely high scores — especially perfect Decision Tree performance — may indicate:

- Potential data leakage  
- Highly separable feature space  
- Risk of overfitting  

In real-world systems, additional validation methods such as cross-validation and external validation datasets should be used to confirm generalization performance.

---

## Project Structure

```text
project-folder/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── model/
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── train_all.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   └── saved_models/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train all models:

```bash
python -m model.train_all
```

3. Run Streamlit app:

```bash
streamlit run app.py
```
