# Bias & Fairness Analysis Report

# 1. Methodology
The bias audit was performed using tumor size as a proxy sensitive attribute. Patients were grouped into "Large Tumors" and "Small Tumors" based on the median value of the `mean area` feature. The model's performance (Recall and Precision) was evaluated separately for each group to check for disparate impact.

# 2. Results
The following table summarizes the model's performance across the subgroups:

| Group | Recall | Precision | Count |
| :--- | :--- | :--- | :--- |
| Large Tumors | 0.9213 | 0.9762 | 284 |
| Small Tumors | 1.0000 | 0.9926 | 285 |

**Disparate Impact Ratio (DIR) = 1.09**

# 3. Analysis
*   **Recall:** The model achieved a 100% recall for small tumors and a 92.1% recall for large tumors. This indicates the model is highly sensitive to early-stage (smaller) tumors, which is clinically beneficial.
*   **Precision:** Both groups maintain precision above 97%, meaning the model rarely produces false positives.
*   **Fairness Assessment:** The Disparate Impact Ratio (DIR) was calculated as 1.09. In fair machine learning standards, a DIR between 0.80 and 1.25 is considered fair. Since 1.09 falls within this accepted range, **no significant disparate impact was detected**. The model performs equitably across both subgroups.

# 4. Mitigation Strategies (If Bias Were Detected)
If a significant bias had been found (e.g., DIR < 0.80), the following mitigation strategies would be implemented:
1.  **Reweighting:** Assign higher weights to the disadvantaged group during model training.
2.  **Threshold Adjustment:** Adjust the decision threshold specifically for the underperforming group to equalize recall.
3.  **Data Augmentation:** Collect more data points for the disadvantaged subgroup.