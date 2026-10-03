# Telco Customer Churn Prediction

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help companies take proactive retention actions.

This project builds a machine learning classification pipeline to predict whether a telecom customer will churn based on customer demographics, account information, services, tenure, and billing details.

The project focuses not only on model accuracy, but also on **precision, recall, F1-score, ROC-AUC, PR-AUC, class imbalance, and classification threshold selection**.

---

## Business Problem

The goal is to predict customers who are likely to churn.

A false negative means a customer who is likely to churn is incorrectly classified as a non-churner. For a retention team, missing these customers can be costly.

Therefore, the project evaluates the trade-off between **precision and recall** rather than relying on accuracy alone.

---

## Dataset

The dataset contains **7,043 customer records and 33 columns**.

The target variable is:

* `Churn Label`

Target distribution:

| Class    | Count | Percentage |
| -------- | ----: | ---------: |
| No Churn | 5,174 |     73.46% |
| Churn    | 1,869 |     26.54% |

The dataset contains numerical and categorical customer attributes such as:

* Tenure
* Monthly Charges
* Total Charges
* Contract
* Payment Method
* Internet Service
* Online Security
* Tech Support
* Paperless Billing
* Senior Citizen
* Partner
* Dependents
* CLTV

---

## Data Cleaning

The following preprocessing steps were performed:

* Removed `Churn Reason` because it is only populated for customers who churned and would introduce target leakage.
* Removed geographic columns:

  * `City`
  * `Zip Code`
  * `Lat Long`
  * `Latitude`
  * `Longitude`
* Converted the target variable:

  * `No → 0`
  * `Yes → 1`
* Converted `Total Charges` from object to numeric.
* Identified blank values in `Total Charges` and handled them appropriately.
* Separated numerical and categorical features.

After feature selection, the dataset contained **20 predictors**.

---

## Train-Test Split

The data was divided into:

* Training set: **5,634 rows**
* Test set: **1,409 rows**

A stratified split was used to preserve the churn/non-churn class distribution.

---

## Preprocessing Pipeline

A `ColumnTransformer` was used to preprocess numerical and categorical features.

### Numerical Features

Numerical variables were scaled using:

```python
StandardScaler()
```

### Categorical Features

Categorical variables were transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing and model steps were combined into a machine learning pipeline to prevent data leakage and keep preprocessing consistent between training and testing.

The preprocessing produced **47 features** after encoding.

---

## Model

### Logistic Regression

Logistic Regression was selected as the initial classification model because it provides a strong and interpretable baseline for binary classification.

Two versions were evaluated:

1. Standard Logistic Regression
2. Logistic Regression with `class_weight="balanced"`

---

## Model Evaluation

Because the dataset is imbalanced, several metrics were considered:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* PR-AUC

### Standard Logistic Regression

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8020 |
| Precision | 0.6426 |
| Recall    | 0.5722 |
| F1-score  | 0.6054 |

Confusion Matrix:

```text
[[916, 119],
 [160, 214]]
```

---

### Class-Weighted Logistic Regression

To give greater importance to the minority churn class, class weighting was applied.

```python
class_weight="balanced"
```

Results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.7431 |
| Precision | 0.5105 |
| Recall    | 0.7834 |
| F1-score  | 0.6181 |
| ROC-AUC   | 0.8489 |
| PR-AUC    | 0.6449 |

Confusion Matrix:

```text
[[754, 281],
 [ 81, 293]]
```

The class-weighted model increased recall for churn customers, while reducing overall accuracy and precision. This demonstrates the trade-off involved when dealing with an imbalanced classification problem.

---

## Classification Threshold Analysis

The default classification threshold of `0.5` is not always appropriate for business problems.

Different thresholds were evaluated to understand the precision-recall trade-off.

| Threshold | Precision | Recall | F1-score |
| --------: | --------: | -----: | -------: |
|      0.20 |     0.406 |  0.971 |    0.572 |
|      0.50 |     0.510 |  0.783 |    0.618 |
|      0.60 |     0.560 |  0.719 |    0.630 |
|      0.70 |     0.607 |  0.620 |    0.614 |
|      0.80 |     0.703 |  0.444 |    0.544 |

Lowering the threshold identifies more potential churners, increasing recall but also increasing false positives.

Increasing the threshold does the opposite, producing fewer positive predictions but missing more actual churners.

This allows the classification threshold to be selected based on the business cost of false positives versus false negatives.

---

## Key Learnings

This project demonstrates several important machine learning concepts:

* Handling categorical and numerical variables
* Feature selection
* Data cleaning
* Train-test splitting
* Stratification
* Feature scaling
* One-hot encoding
* Scikit-learn pipelines
* Preventing data leakage
* Imbalanced classification
* Class weighting
* Confusion matrix interpretation
* Precision vs. recall trade-offs
* ROC-AUC
* PR-AUC
* Classification threshold tuning

A major takeaway is that **accuracy alone is not sufficient for evaluating an imbalanced classification problem**.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* OpenPyXL

---

## Project Structure

```text
Telco-Customer-Churn/
│
├── project.py
├── requirements.txt
├── README.md
├── .gitignore
└── Telco_customer_churn.xlsx
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Telco-Customer-Churn
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the project

```bash
python project.py
```

---

## Conclusion

This project demonstrates an end-to-end approach to customer churn prediction, from data cleaning and preprocessing to model evaluation and threshold analysis.

Rather than treating model accuracy as the only objective, the project evaluates how different modeling decisions affect the identification of customers who are likely to churn.

This provides a more practical approach to applying machine learning to customer retention problems.
