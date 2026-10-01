import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    cross_val_predict
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

from sklearn.inspection import permutation_importance


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_excel("Telco_customer_churn.xlsx")

print("Original shape:", df.shape)
print(df.info())


# ============================================================
# 2. DATA CLEANING
# ============================================================

df['Total Charges'] = pd.to_numeric(
    df['Total Charges'],
    errors='coerce'
)

df['Total Charges'] = df['Total Charges'].fillna(0)

df = df.drop(columns=['Churn Reason'])

drop_cols = [
    'CustomerID',
    'Count',
    'Country',
    'State',
    'Churn Value',
    'Churn Score'
]

df = df.drop(columns=drop_cols)

print("\nShape after cleaning:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=['Churn Label'])

y = df['Churn Label'].map({
    'No': 0,
    'Yes': 1
})

geo_cols = [
    'City',
    'Zip Code',
    'Lat Long',
    'Latitude',
    'Longitude'
]

X = X.drop(columns=geo_cols)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)

print("\nRemaining features:")
print(X.columns.tolist())


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_feature = X_train.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

categorical_feature = X_train.select_dtypes(
    include='object'
).columns.tolist()

print("\nNumeric features:")
print(numeric_feature)

print("\nCategorical features:")
print(categorical_feature)

preprocesser = ColumnTransformer(
    transformers=[
        (
            'num',
            StandardScaler(),
            numeric_feature
        ),
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_feature
        )
    ]
)


# ============================================================
# 6. BASELINE MODEL COMPARISON
# ============================================================

logreg_pipeline = Pipeline([
    (
        'preprocessor',
        preprocesser
    ),
    (
        'model',
        LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    )
])

tree_pipeline = Pipeline([
    (
        'preprocessor',
        preprocesser
    ),
    (
        'model',
        DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        )
    )
])

rf_pipeline = Pipeline([
    (
        'preprocessor',
        preprocesser
    ),
    (
        'model',
        RandomForestClassifier(
            class_weight='balanced',
            n_estimators=200,
            random_state=42
        )
    )
])

svm_pipeline = Pipeline([
    (
        'preprocessor',
        preprocesser
    ),
    (
        'model',
        SVC(
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    )
])

gb_pipeline = Pipeline([
    (
        'preprocessor',
        preprocesser
    ),
    (
        'model',
        GradientBoostingClassifier(
            random_state=42
        )
    )
])

models = {
    'Logistic Regression': logreg_pipeline,
    'Decision Tree': tree_pipeline,
    'Random Forest': rf_pipeline,
    'SVM': svm_pipeline,
    'Gradient Boosting': gb_pipeline
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'PR-AUC': average_precision_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)

print("\nBaseline Model Comparison:")
print(
    results_df
    .sort_values('F1', ascending=False)
    .to_string(index=False)
)


# ============================================================
# 7. STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision'
}

cv_results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({
        'Model': name,
        'F1 Mean': scores['test_f1'].mean(),
        'F1 Std': scores['test_f1'].std(),
        'ROC-AUC Mean': scores['test_roc_auc'].mean(),
        'ROC-AUC Std': scores['test_roc_auc'].std(),
        'PR-AUC Mean': scores['test_pr_auc'].mean(),
        'PR-AUC Std': scores['test_pr_auc'].std()
    })

cv_results_df = pd.DataFrame(cv_results)

print("\n5-Fold Cross-Validation:")
print(
    cv_results_df
    .sort_values('F1 Mean', ascending=False)
    .to_string(index=False)
)


# ============================================================
# 8. HYPERPARAMETER TUNING
# ============================================================

logreg_param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100]
}

logreg_grid = GridSearchCV(
    logreg_pipeline,
    param_grid=logreg_param_grid,
    scoring=scoring,
    refit='f1',
    cv=cv,
    n_jobs=-1
)

logreg_grid.fit(X_train, y_train)

print("\nLogistic Regression")
print("Best parameters:", logreg_grid.best_params_)
print("Best CV F1:", logreg_grid.best_score_)


svm_param_grid = {
    'model__C': [0.1, 1, 10],
    'model__gamma': ['scale', 0.01, 0.1]
}

svm_grid = GridSearchCV(
    svm_pipeline,
    param_grid=svm_param_grid,
    scoring=scoring,
    refit='f1',
    cv=cv,
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("\nSVM")
print("Best parameters:", svm_grid.best_params_)
print("Best CV F1:", svm_grid.best_score_)


gb_param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [1, 2, 3]
}

gb_grid = GridSearchCV(
    gb_pipeline,
    param_grid=gb_param_grid,
    scoring=scoring,
    refit='f1',
    cv=cv,
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

print("\nGradient Boosting")
print("Best parameters:", gb_grid.best_params_)
print("Best CV F1:", gb_grid.best_score_)


rf_param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_leaf': [1, 2, 5]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid=rf_param_grid,
    scoring=scoring,
    refit='f1',
    cv=cv,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nRandom Forest")
print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", rf_grid.best_score_)


# ============================================================
# 9. THRESHOLD SELECTION USING OOF PREDICTIONS
# ============================================================

rf_oof_prob = cross_val_predict(
    rf_grid.best_estimator_,
    X_train,
    y_train,
    cv=cv,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

threshold_results = []

for threshold in np.arange(0.30, 0.71, 0.01):

    y_pred = (
        rf_oof_prob >= threshold
    ).astype(int)

    threshold_results.append({
        'Threshold': threshold,
        'Precision': precision_score(y_train, y_pred),
        'Recall': recall_score(y_train, y_pred),
        'F1': f1_score(y_train, y_pred)
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nBest OOF Thresholds:")
print(
    threshold_df
    .sort_values('F1', ascending=False)
    .head(10)
    .to_string(index=False)
)

best_threshold = (
    threshold_df
    .loc[threshold_df['F1'].idxmax(), 'Threshold']
)

print("\nSelected threshold:", best_threshold)


# ============================================================
# 10. FINAL RANDOM FOREST EVALUATION
# ============================================================

final_rf = rf_grid.best_estimator_

final_rf.fit(X_train, y_train)

y_test_prob = final_rf.predict_proba(X_test)[:, 1]

y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)

print("\nFinal Random Forest Performance:")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall   :", recall_score(y_test, y_test_pred))
print("F1       :", f1_score(y_test, y_test_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_test_prob))
print("PR-AUC   :", average_precision_score(y_test, y_test_prob))


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=['No Churn', 'Churn']
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)

disp.plot()

plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.show()


# ============================================================
# 13. FINAL MODEL COMPARISON
# ============================================================

tuned_models = {
    'Logistic Regression': logreg_grid.best_estimator_,
    'SVM': svm_grid.best_estimator_,
    'Gradient Boosting': gb_grid.best_estimator_,
    'Random Forest': rf_grid.best_estimator_
}

final_results = []

for name, model in tuned_models.items():

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.50).astype(int)

    final_results.append({
        'Model': name,
        'Threshold': 0.50,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, prob),
        'PR-AUC': average_precision_score(y_test, prob)
    })

final_results_df = pd.DataFrame(final_results)

print("\nTuned Model Comparison at Default Threshold (0.50):")
print(
    final_results_df
    .sort_values('F1', ascending=False)
    .to_string(index=False)
)

# ============================================================
# 14. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

preprocesser = final_rf.named_steps['preprocessor']
rf_model = final_rf.named_steps['model']

feature_names = (
    preprocesser
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    'Importance',
    ascending=False
)

print("\nRandom Forest Feature Importance:")
print(
    importance_df
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 15. PERMUTATION IMPORTANCE
# ============================================================

perm = permutation_importance(
    final_rf,
    X_test,
    y_test,
    scoring='f1',
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance Mean': perm.importances_mean,
    'Importance Std': perm.importances_std
})

perm_df = perm_df.sort_values(
    'Importance Mean',
    ascending=False
)

print("\nPermutation Importance:")
print(
    perm_df
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 16. PERMUTATION IMPORTANCE PLOT
# ============================================================

top_perm = (
    perm_df
    .head(10)
    .sort_values('Importance Mean')
)

plt.figure(figsize=(8, 5))

plt.barh(
    top_perm['Feature'],
    top_perm['Importance Mean'],
    xerr=top_perm['Importance Std']
)

plt.xlabel('Mean decrease in F1')
plt.ylabel('Feature')
plt.title('Random Forest Permutation Feature Importance')

plt.tight_layout()
plt.show()


# ============================================================
# 17. SAVE FINAL RESULTS
# ============================================================

final_results_df.to_csv(
    'final_model_comparison.csv',
    index=False
)

print("\nFinal model comparison saved.")