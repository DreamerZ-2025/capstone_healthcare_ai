import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from src.data_preprocessing import load_and_preprocess_data

def train_models():
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()

    # Define models and hyperparameters
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        }
    }

    best_model = None
    best_auc = 0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, config in models.items():
        print(f"Training {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        y_prob = grid.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"{name} Best Params: {grid.best_params_}")
        print(f"{name} ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        if auc > best_auc:
            best_auc = auc
            best_model = grid.best_estimator_

    # Save the best model
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"Saved best model with ROC-AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_models()