import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

df = pd.read_csv('AIML Dataset.csv')

if 'isFraud' in df.columns:
    df.dropna(subset=['isFraud'], inplace=True)

cols_to_drop = [c for c in ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step'] if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
    df['balenceDiffOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
    df['balenceDiffDest'] = df['oldbalanceDest'] - df['newbalanceDest']

if 'isFraud' not in df.columns:
    raise ValueError('isFraud column not found in dataset')

categorical = ['type'] if 'type' in df.columns else []
numeric = [c for c in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'] if c in df.columns]

X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ],
    remainder='drop'
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', xgb)
])

print('Training XGBoost pipeline...')
pipeline.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = pipeline.predict(X_test)
print('\nEvaluation on test set:')
print(classification_report(y_test, y_pred, digits=4))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))

joblib.dump(pipeline, 'Fraud_detection_pipeline.pkl')
print('\nSaved pipeline to Fraud_detection_pipeline.pkl')

