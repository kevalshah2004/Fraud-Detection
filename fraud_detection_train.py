import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, classification_report, 
                             confusion_matrix, accuracy_score)

warnings.filterwarnings('ignore')

def run_fraud_detection():
    # 1. Load Data
    df = pd.read_csv('AIML Dataset.csv')

    # 2. Cleaning & Feature Engineering
    if 'isFraud' in df.columns:
        df.dropna(subset=['isFraud'], inplace=True)

    cols_to_drop = [c for c in ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Creating balance difference features
    for prefix in ['Org', 'Dest']:
        old_col, new_col = f'oldbalance{prefix}', f'newbalance{prefix}'
        if old_col in df.columns and new_col in df.columns:
            df[f'balenceDiff{prefix}'] = df[old_col] - df[new_col]

    # New fractional change feature
    epsilon = 1e-9 
    df['fractional_change_org'] = (df['oldbalanceOrg'] - df['newbalanceOrig']) / (df['oldbalanceOrg'] + epsilon)

    # 3. Defining Features
    categorical = ['type'] if 'type' in df.columns else []
    numeric = [c for c in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
                           'newbalanceDest', 'balenceDiffOrg', 'balenceDiffDest', 'fractional_change_org'] if c in df.columns]

    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 4. Pipeline Setup
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(drop='first'), categorical)
        ],
        remainder='drop'
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # XGBoost internal parallelization is safe on Windows
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        n_jobs=-1, 
        random_state=42
    )

    pipeline = Pipeline([('prep', preprocessor), ('clf', xgb)])

    # 5. CROSS VALIDATION - CRITICAL CHANGE FOR WINDOWS
    print('Running Stratified 5-Fold Cross-Validation...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # n_jobs must be 1 here to avoid the _posixsubprocess error
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='average_precision', n_jobs=1)

    print(f'Cross-Validated AUPRC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
    print('-' * 30)

    # 6. Final Fit and Metrics
    print('Training final XGBoost pipeline...')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    print('\nEvaluation on test set:')
    print(classification_report(y_test, y_pred, digits=4))
    print(f'Test AUPRC: {average_precision_score(y_test, y_probs):.4f}')

    # 7. Visualization
    plt.figure(figsize=(10,8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.show()

    joblib.dump(pipeline, 'Fraud_detection_pipeline.pkl')
    print('\nSaved pipeline to Fraud_detection_pipeline.pkl')

if __name__ == "__main__":
    run_fraud_detection()