# Machine Learning Report: Electrical Grid Stability Prediction

**Module:** IT5022 - Fundamentals of Machine Learning  
**Assignment:** Group Assignment 01  
**Dataset:** Electrical Grid Stability and Control Simulation Dataset  
**Baseline Majority Accuracy:** 63.80%

---

## 1. Executive Summary & Experimental Results

This study develops and evaluates non-deep-learning supervised classification models to predict decentralized smart electrical grid stability (`stabf`: `stable` vs. `unstable`) from 12 operational features (reaction times $\tau_1-\tau_4$, nominal powers $p_1-p_4$, and price elasticity coefficients $g_1-g_4$).

To avoid critical **target leakage**, the eigenvalue continuous score `stab` was excluded from the feature space, as `stabf` is deterministically derived from it ($\text{stab} \le 0 \implies \text{stable}$).

### Experimental Results on Held-Out Test Set ($N=2,000$):

| Metric | Zero-Rule Baseline | Logistic Regression (Pipeline) | Random Forest ($n=200$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 63.80% | **82.00%** | **92.90%** |
| **Macro F1-Score** | 0.3895 | **0.8009** | **0.9220** |
| **Weighted F1-Score** | 0.4970 | **0.8179** | **0.9285** |
| **Stable Precision** | 0.00% | 77.74% | **92.92%** |
| **Stable Recall** | 0.00% | 70.44% | **87.02%** |
| **Stable F1-Score** | 0.00% | 73.91% | **89.87%** |
| **Unstable Precision** | 63.80% | 84.08% | **92.89%** |
| **Unstable Recall** | 100.00% | 88.56% | **96.24%** |
| **Unstable F1-Score** | 77.90% | 86.26% | **94.53%** |

---

## 2. Key Findings & Discussion

1. **Superiority of Non-Linear Ensemble Modeling:**  
   Random Forest achieved an overall accuracy of **92.90%** and Macro-F1 of **0.9220**, significantly outperforming both the zero-rule baseline (63.80%) and Logistic Regression (82.00%). This suggests that grid stability depends on non-linear interactions between reaction times and elasticity parameters. Section 3 shows that supplying those interactions explicitly makes Logistic Regression the best model.

2. **Critical Operational Risk (False Stables):**  
   In electrical grid management, a **False Stable** error (predicting a grid is safe/stable when it is actually unstable) is catastrophic as it risks cascading blackouts.
   - Logistic Regression misclassified **146 unstable states as stable** (recall of 88.56%).
   - Random Forest minimized this critical error to only **48 unstable states** (recall of 96.24%).

3. **Feature Importance (MDI):**  
   Feature analysis identified participant reaction times ($\tau_1-\tau_4$) as the most influential predictors of stability, followed by participant price elasticity coefficients ($g_1-g_4$). Nominal power variables ($p_1-p_4$) had comparatively lower individual importance due to the closed net power balance constraint ($\sum p = 0$).

4. **Academic Limitations:**  
   The dataset represents simulated differential equation responses in a 4-node star topology. These results do not guarantee real-world safety on dynamic physical grids with complex topologies, real-time sensor noise, or variable renewable energy integration.

---

## 3. Accuracy-Improvement Experiments (5-fold CV on training set; test set used once)

| Model | Accuracy | Macro-F1 | ROC-AUC | Missed unstable | False alarms |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression + degree-2 polynomial features (C = 10) | **95.55%** | **0.9518** | **0.9897** | 45 | 44 |
| Random Forest tuned (max_depth = 20) | 92.70% | 0.9201 | 0.9829 | 55 | 91 |
| LR polynomial, recall-oriented threshold 0.25 | 94.05% | 0.9338 | 0.9897 | **17** | 102 |

- Dropping the redundant `p1` (exactly `-(p2+p3+p4)`) does not change LR and slightly helps RF; power features alone are no better than the majority baseline.
- Tuning Random Forest changed CV macro-F1 by < 0.01; interaction features raised LR CV macro-F1 from 0.796 to 0.934.
- 61% of Random Forest errors have |stab| < 0.01 (vs 14% of all test records): errors concentrate at the stability boundary.

The full report is `report/IT5022_Fundamentals_of_Machine_Learning_Assignment_01.pdf` (editable source: `report/report.docx`).

## 4. Generated Figures & Output References
- Target Distribution: `outputs/figures/target_distribution.png`
- Leakage Demonstration Plot: `outputs/figures/stab_vs_stabf_leakage.png`
- 12-Feature Correlation Heatmap: `outputs/figures/feature_correlation_heatmap.png`
- Comparative Confusion Matrices: `outputs/figures/confusion_matrices.png`
- Feature Importance Bar Chart: `outputs/figures/rf_feature_importance.png`
- Metric Summary Table: `outputs/metrics/model_comparison.csv`
- Final comparison of all models: `outputs/metrics/final_model_comparison.csv`
- ROC curves, PCA, permutation importance, threshold trade-off: `outputs/figures/`
