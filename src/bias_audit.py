import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import recall_score, precision_score

def audit_bias():
    # Load data and model
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Create a proxy sensitive group (e.g., 'Large Tumor' vs 'Small Tumor')
    # We use 'mean area' as a proxy for tumor size
    median_area = X['mean area'].median()
    X['tumor_size_group'] = X['mean area'].apply(lambda x: 'Large' if x > median_area else 'Small')
    
    # Make predictions
    X_scaled = scaler.transform(X.drop('tumor_size_group', axis=1))
    y_pred = model.predict(X_scaled)
    
    # Evaluate per group
    results = []
    for group in ['Large', 'Small']:
        mask = X['tumor_size_group'] == group
        recall = recall_score(y[mask], y_pred[mask])
        precision = precision_score(y[mask], y_pred[mask])
        results.append({'Group': group, 'Recall': recall, 'Precision': precision, 'Count': mask.sum()})
    
    df_results = pd.DataFrame(results)
    print("Bias Audit Results (Proxy: Tumor Size):")
    print(df_results)
    
    # Check for Disparate Impact
    # Disparate Impact Ratio = (Recall of unprivileged) / (Recall of privileged)
    # A ratio < 0.8 indicates potential bias.
    di_ratio = df_results.loc[df_results['Group'] == 'Small', 'Recall'].values[0] / df_results.loc[df_results['Group'] == 'Large', 'Recall'].values[0]
    print(f"Disparate Impact Ratio (Small vs Large): {di_ratio:.2f}")
    if di_ratio < 0.8:
        print("WARNING: Potential bias detected. Model performs significantly worse for smaller tumors.")
    else:
        print("No significant disparate impact detected based on this proxy.")

if __name__ == "__main__":
    audit_bias()