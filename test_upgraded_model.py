import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 1: Dependencies loaded OK")

# ── Step 2: Data Loading ──────────────────────────────────────
heart_data = pd.read_csv('data.csv')
print(f"\nSTEP 2: Dataset shape: {heart_data.shape}")
print(f"Missing values: {heart_data.isnull().sum().sum()}")
print(f"Class distribution:\n{heart_data['target'].value_counts()}")

# ── Step 3: Feature Engineering ──────────────────────────────
X = heart_data.drop(columns='target')
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nSTEP 3: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")

# ── Step 4: Multiple Models ───────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, Y_train)
    y_pred       = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    results[name] = {
        'model':     model,
        'accuracy':  accuracy_score(Y_test, y_pred),
        'precision': precision_score(Y_test, y_pred),
        'recall':    recall_score(Y_test, y_pred),
        'f1':        f1_score(Y_test, y_pred),
        'roc_auc':   roc_auc_score(Y_test, y_pred_proba),
    }

results_df = pd.DataFrame(results).T
print("\nSTEP 4: Model Comparison:")
print(results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].round(4))

# ── Step 5: Hyperparameter Tuning ────────────────────────────
param_grid = {
    'n_estimators':    [100, 200],
    'max_depth':       [10, 20, None],
    'min_samples_split': [2, 5],
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_scaled, Y_train)
best_rf = rf_grid.best_estimator_
print(f"\nSTEP 5: Best RF params: {rf_grid.best_params_}")
print(f"Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

# ── Step 6: Stacking Ensemble ─────────────────────────────────
estimators = [
    ('rf',  best_rf),
    ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
]
stacking_model = StackingClassifier(estimators=estimators,
                                    final_estimator=LogisticRegression(max_iter=1000), cv=5)
stacking_model.fit(X_train_scaled, Y_train)
y_pred_stack       = stacking_model.predict(X_test_scaled)
y_pred_proba_stack = stacking_model.predict_proba(X_test_scaled)[:, 1]

print(f"\nSTEP 6: Stacking Ensemble:")
print(f"  Accuracy : {accuracy_score(Y_test, y_pred_stack):.4f}")
print(f"  Precision: {precision_score(Y_test, y_pred_stack):.4f}")
print(f"  Recall   : {recall_score(Y_test, y_pred_stack):.4f}")
print(f"  F1-Score : {f1_score(Y_test, y_pred_stack):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(Y_test, y_pred_proba_stack):.4f}")

# ── Step 7: Neural Network ────────────────────────────────────
nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1,  activation='sigmoid'),
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train_scaled, Y_train, epochs=100, batch_size=32,
                       validation_split=0.2, verbose=0)

y_pred_nn       = (nn_model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()
y_pred_proba_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()

print(f"\nSTEP 7: Neural Network:")
print(f"  Accuracy : {accuracy_score(Y_test, y_pred_nn):.4f}")
print(f"  Precision: {precision_score(Y_test, y_pred_nn):.4f}")
print(f"  Recall   : {recall_score(Y_test, y_pred_nn):.4f}")
print(f"  F1-Score : {f1_score(Y_test, y_pred_nn):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(Y_test, y_pred_proba_nn):.4f}")

# ── Step 8: Feature Importance ────────────────────────────────
feature_importance = pd.DataFrame({
    'feature':    X.columns,
    'importance': best_rf.feature_importances_,
}).sort_values('importance', ascending=False)
print(f"\nSTEP 8: Top 5 Features:\n{feature_importance.head()}")

# ── Step 9: Confusion Matrix ──────────────────────────────────
print(f"\nSTEP 9: Classification Report (Stacking):")
print(classification_report(Y_test, y_pred_stack, target_names=['No Disease', 'Disease']))

# ── Step 10: Model Persistence ────────────────────────────────
joblib.dump(stacking_model, 'heart_disease_stacking_model.pkl')
joblib.dump(scaler,         'heart_disease_scaler.pkl')
nn_model.save('heart_disease_nn_model.h5')
print("STEP 10: Models saved OK")

# ── Step 11: Prediction Pipeline ─────────────────────────────
def predict_heart_disease(input_data, model_type='stacking'):
    input_scaled = scaler.transform(np.asarray(input_data).reshape(1, -1))
    if model_type == 'stacking':
        prediction  = stacking_model.predict(input_scaled)[0]
        probability = stacking_model.predict_proba(input_scaled)[0]
        confidence  = probability[1]
    else:
        prediction  = (nn_model.predict(input_scaled, verbose=0) > 0.5).astype(int)[0][0]
        confidence  = nn_model.predict(input_scaled, verbose=0)[0][0]
    return ("Heart Disease" if prediction == 1 else "No Heart Disease"), confidence

print("\nSTEP 11: Prediction Pipeline Tests:")
for label, inp in [
    ("Patient 1 (low risk)", (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2)),
    ("Patient 2 (high risk)", (51, 0, 0, 130, 305, 0, 1, 142, 1, 1.2, 1, 0, 3)),
]:
    res, conf = predict_heart_disease(inp, 'stacking')
    print(f"  {label}: {res} ({conf:.2%})")

# ── Step 12: Final Comparison ─────────────────────────────────
final = pd.DataFrame({
    'Model':   ['Logistic Regression', 'Random Forest', 'Gradient Boosting',
                'SVM', 'Stacking Ensemble', 'Neural Network'],
    'Accuracy': [results['Logistic Regression']['accuracy'], results['Random Forest']['accuracy'],
                 results['Gradient Boosting']['accuracy'],   results['SVM']['accuracy'],
                 accuracy_score(Y_test, y_pred_stack),       accuracy_score(Y_test, y_pred_nn)],
    'ROC-AUC':  [results['Logistic Regression']['roc_auc'],  results['Random Forest']['roc_auc'],
                 results['Gradient Boosting']['roc_auc'],    results['SVM']['roc_auc'],
                 roc_auc_score(Y_test, y_pred_proba_stack),  roc_auc_score(Y_test, y_pred_proba_nn)],
})
print(f"\nSTEP 12: Final Comparison:\n{final.round(4)}")
print(f"\nBest Model : {final.loc[final['ROC-AUC'].idxmax(), 'Model']}")
print(f"Best ROC-AUC: {final['ROC-AUC'].max():.4f}")
print("\n" + "=" * 60)
print("ALL STEPS PASSED - OK")
