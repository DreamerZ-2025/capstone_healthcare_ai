import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    # Load built-in dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target') # 0 = Malignant, 1 = Benign

    # Split data (80/20) with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (crucial for models like Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for deployment
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")