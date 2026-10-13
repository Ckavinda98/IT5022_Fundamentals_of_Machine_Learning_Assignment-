# Machine Learning-Based Prediction of Electrical Grid Stability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Evaluation-lightgrey.svg)](#)

This repository contains an end-to-end Machine Learning pipeline developed for **IT5022 - Fundamentals of Machine Learning (Group Assignment 01)** to predict the stability of a decentralized 4-node smart electrical grid.

---

## 📌 Project Overview & Research Question

**Research Question:**  
*Can operating parameters of a simulated electrical grid (participant reaction times, power flows, and price elasticity) reliably predict whether an operating state is `stable` or `unstable`?*

### Dataset Specifications
- **Source:** Simulated Electrical Grid Stability and Control Dataset ([Kaggle Source](https://www.kaggle.com/datasets/avikdas567/electrical-grid-stability-and-control-simulation)).
- **Size:** 10,000 observations × 14 variables.
- **Input Features ($X$):** 12 independent operating features:
  - `tau1` to `tau4`: Participant reaction times (seconds).
  - `p1` to `p4`: Power produced (positive, `p1`) and power consumed (negative, `p2`, `p3`, `p4`).
  - `g1` to `g4`: Price elasticity coefficients ($\gamma$).
- **Target Label ($y$):** `stabf` (categorical: `stable` vs. `unstable`).
  - **Class Distribution:** 6,380 `unstable` (63.8%) | 3,620 `stable` (36.2%).
  - **Majority Baseline Accuracy:** **63.80%**.

### ⚠️ Target Leakage Prevention
The raw dataset contains a continuous score `stab` representing the maximal real part of characteristic differential equation roots. The categorical target `stabf` is deterministically derived from `stab` ($\text{stab} \le 0 \implies \text{stable}$, $\text{stab} > 0 \implies \text{unstable}$).  
**To maintain rigorous experimental integrity and prevent fatal target leakage, `stab` is strictly dropped from the feature space $X$.**

---

## 📂 Project Directory Structure

```text
IT5022_Fundamentals_of_Machine_Learning_Assignment-/
├── README.md                           # Comprehensive documentation and run guide
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Ignored files (virtual environments, checkpoints)
├── members.txt                         # Group member details
├── submission.txt                      # Submission links (GitHub, Dataset, Video)
├── data/
│   ├── README.md                       # Data schema & attribute documentation
│   └── raw/
│       └── electrical_grid_stability.csv # 10,000 samples dataset
├── notebooks/                          # Interactive analysis notebooks
├── src/                                # Modular production-ready Python source code
│   ├── __init__.py
│   ├── data_checks.py                  # Data quality & schema verification
│   ├── preprocessing.py               # Leakage-safe stratified train/test split
│   ├── train_models.py                # Logistic Regression & Random Forest pipelines
│   ├── train_predictor.py              # Train and save the UI prediction model
│   ├── evaluation.py                  # Metrics & confusion matrix calculations
│   └── run_pipeline.py                # End-to-end execution runner
├── app.py                              # Interactive Streamlit prediction UI
├── outputs/
│   ├── figures/                        # Generated high-resolution plots
│   │   ├── target_distribution.png
│   │   ├── stab_vs_stabf_leakage.png
│   │   ├── feature_correlation_heatmap.png
│   │   ├── confusion_matrices.png
│   │   └── rf_feature_importance.png
│   └── metrics/                        # Quantitative metric evaluations
│       └── model_comparison.csv
└── report/
    ├── report_draft.md                 # Academic report summary and findings
    └── report.pdf
```

---

## 🚀 Step-by-Step Setup and Execution Guide

Follow these steps to clone, set up the virtual environment, install dependencies, and run the pipeline.

### Step 1: Clone the Repository
Open your terminal (PowerShell, Bash, or Command Prompt) and run:
```bash
git clone https://github.com/Ckavinda98/IT5022_Fundamentals_of_Machine_Learning_Assignment-.git
cd IT5022_Fundamentals_of_Machine_Learning_Assignment-
```

### Step 2: Create a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If you encounter execution policy restrictions in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first).*

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
Upgrade `pip` and install all necessary scientific libraries:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Complete Machine Learning Pipeline
Execute the automated end-to-end training and evaluation pipeline in one command:
```bash
python -m src.run_pipeline
```

This single command will automatically:
1. Load and stratify-split the dataset (8,000 train / 2,000 test) while excluding `stab`.
2. Generate all exploratory data analysis figures in `outputs/figures/`.
3. Train the **Logistic Regression** baseline (with standard scaling in a pipeline).
4. Train the **Random Forest Classifier** (200 estimators).
5. Evaluate both models on the held-out test data and save metrics to `outputs/metrics/model_comparison.csv`.
6. Generate side-by-side confusion matrices and feature importance rankings.

### Step 5: Train a Model and Open the Prediction UI

After installing the dependencies, train and save the Random Forest model once,
then start the Streamlit interface:

```powershell
python -m src.train_predictor
streamlit run app.py
```

The app opens in your browser. Enter `tau1`-`tau4`, `p1`-`p4`, and `g1`-`g4`,
then select **Predict stability**. It displays the predicted class and the
model-estimated probabilities for `stable` and `unstable`. Power values must
approximately satisfy `p1 + p2 + p3 + p4 = 0`. The trained model is saved under
`outputs/models/`; retrain it after changing the dataset or model settings.

### Deploy the UI so others can use it

The app can be deployed to Streamlit Community Cloud and shared through its
public URL. Push the project branch to GitHub:

```powershell
git push -u origin Development-Darshana-Nadeera
```

Then sign in at <https://share.streamlit.io>, connect GitHub, and create an app
using this repository, the `Development-Darshana-Nadeera` branch, and `app.py`
as the entrypoint. Set the app's access to public, deploy it, then share the
generated `*.streamlit.app` URL. The app trains and caches its model from the
CSV on its first start if the locally generated model artifact is absent.

### Step 6: (Optional) Run Individual Modules
You can also run individual modular components directly:
- **Audit Data Quality:**
  ```bash
  python -m src.data_checks
  ```
- **Check Data Splitting:**
  ```bash
  python -m src.preprocessing
  ```

---

## 📊 Experimental Results

Both models were evaluated on the exact same 2,000 held-out test records using stratified splitting ($80/20$ ratio, `random_state=42`):

| Evaluation Metric | Zero-Rule Baseline | Logistic Regression (Pipeline) | Random Forest ($n=200$) |
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

### Key Discussion Points:
1. **Non-linear Model Superiority:** The Random Forest achieved **92.90% accuracy**, demonstrating that stability is governed by non-linear parameter interactions.
2. **Safety Risk Analysis (False Stables):** In power systems, classifying an unstable grid as stable can trigger catastrophic outages. Random Forest minimized false-stable classifications to 48 cases, achieving a **96.24% unstable recall**.
3. **Feature Importance:** Analysis shows participant reaction times ($\tau_1-\tau_4$) and price elasticity ($\gamma$) are the strongest indicators of stability.

---

## 👥 Contributors & Academic Integrity
- Developed for **IT5022 - Fundamentals of Machine Learning**.
- Refer to `members.txt` for group contributor details.
- Licensed for academic evaluation and research.
