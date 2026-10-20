# 🩺 Breast Cancer Diagnostic Classification: Multi-Model Benchmark
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Wisconsin%20Diagnostic-critical.svg)](#)
[![Accuracy](https://img.shields.io/badge/Best%20Accuracy-98.25%25-brightgreen.svg)](#)
[![Recall](https://img.shields.io/badge/Recall%20(Sensitivity)-100%25-success.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
An end-to-end clinical machine learning study benchmarking tree-based and ensemble architectures on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. 
The project implements exploratory feature correlation heatmaps, automated feature scaling via Scikit-Learn `Pipeline`, and head-to-head performance evaluation across six competitive algorithms (*Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Hist Gradient Boosting*), with particular focus on clinical sensitivity (**Recall**).
---
## 📌 Table of Contents
- [Clinical Problem & Motivation](#-clinical-problem--motivation)
- [Dataset Characteristics](#-dataset-characteristics)
- [Pipeline Architecture](#-pipeline-architecture)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Models Evaluated](#-models-evaluated)
- [Benchmark Results & Clinical Insights](#-benchmark-results--clinical-insights)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Future Roadmap](#-future-roadmap)
---
## 🏥 Clinical Problem & Motivation
Breast cancer remains one of the most common cancers diagnosed among women worldwide. Early and accurate detection drastically reduces mortality rates by enabling timely intervention. 
In clinical diagnostics, **classification trade-offs have real-world consequences**:
- **False Positive (FP)**: Diagnosing a benign mass as malignant causes severe patient anxiety and unnecessary invasive biopsies.
- **False Negative (FN)**: Diagnosing a malignant tumor as benign delays life-saving therapy.
Therefore, models must be evaluated not only on raw **Accuracy**, but primarily on **Recall (Sensitivity)** and **ROC-AUC**.
---
## 🔬 Dataset Characteristics
The benchmark utilizes the canonical **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset:
- **Total Patients / Samples**: 569 instances
- **Class Distribution**: 
  - `Malignant (0)`: 212 cases (~37.3%)
  - `Benign (1)`: 357 cases (~62.7%)
- **Feature Attributes**: 30 real-valued continuous features computed from digitized images of Fine Needle Aspirates (FNA) of breast masses, covering three dimensions (Mean, Standard Error, Worst/Largest) for 10 cell nuclear characteristics:
  - Radius, Texture, Perimeter, Area, Smoothness
  - Compactness, Concavity, Concave points, Symmetry, Fractal dimension
---
## 🏗️ Pipeline Architecture
```text
Wisconsin Diagnostic Breast Cancer Dataset (569 samples, 30 features)
                              │
                              ▼
        [Exploratory Data Analysis & Collinearity Matrix]
            - Target frequency distribution audit
            - 12x10 Pearson Correlation Heatmap
                              │
                              ▼
        [Stratified Train/Test Split (80% Train / 20% Test, seed=42)]
                              │
                              ▼
        [Scikit-Learn Pipeline: StandardScaler() + Estimator]
            (Prevents data leakage across test partitions)
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                 Multi-Model Benchmark Tournament             │
  ├──────────────────────────────┬───────────────────────────────┤
  │ 1. Decision Tree             │ 4. AdaBoost Classifier        │
  │ 2. Random Forest             │ 5. Gradient Boosting          │
  │ 3. Extra Trees               │ 6. Hist Gradient Boosting     │
  └──────────────────────────────┴───────────────────────────────┘
                              │
                              ▼
        [Multi-Metric Evaluation & Clinical Auditing]
    (Accuracy, Precision, Recall / Sensitivity, F1, ROC-AUC)
                              │
                              ▼
        [Comparative Visualization: Accuracy Barplot]

  

  
📊 Exploratory Data Analysis (EDA)

  

  
Class Distribution Check: Audits the ratio between benign and malignant diagnoses to assess label distribution prior to splitting.

  
Correlation Matrix (Heatmap):
  
  

  
A 
12
×
10
12×10 Seaborn heatmap visualizes pairwise Pearson correlation coefficients (
𝑟
r).

  
Highlights strong collinearity among geometric properties (e.g., radius_mean, perimeter_mean, and area_mean exhibit 
𝑟
>
0.95
r>0.95).

  
Validates the decision to incorporate tree ensembles, which are robust to multi-collinear inputs.

  

  

  

  
⚙️ Models Evaluated

  

All models are trained inside an automated pipeline encapsulating StandardScaler():


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
Estimator	Family	Strategy / Mechanism
Decision Tree	Single Tree	Baseline greedy partitioner based on Gini impurity.
Random Forest	Bagging Ensemble	Constructs randomized decision trees over bootstrapped feature subsets.
Extra Trees	Extremely Randomized	Randomizes split thresholds to further suppress model variance.
AdaBoost	Sequential Boosting	Adaptively weights previously misclassified instances.
Gradient Boosting	Loss Optimization	Fits sequential trees to the negative pseudo-residuals of the loss function.
Hist Gradient Boosting	Binned Gradient Boosting	Bins continuous features into integer bins (similar to LightGBM), accelerating training and smoothing boundaries.

  

  
📈 Benchmark Results & Clinical Insights

  
Performance Comparison Table

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
Model Architecture	Accuracy	Precision	Recall (Sensitivity)	F1-Score	ROC-AUC
🥇 Hist Gradient Boosting	98.25%	0.9726	1.0000	0.9861	0.9984
🥈 Extra Trees Classifier	97.37%	0.9722	0.9859	0.9790	0.9941
🥉 AdaBoost Classifier	97.37%	0.9722	0.9859	0.9790	0.9892
Random Forest	96.49%	0.9589	0.9859	0.9722	0.9931
Gradient Boosting	95.61%	0.9583	0.9718	0.9650	0.9908
Decision Tree (Baseline)	93.86%	0.9315	0.9577	0.9444	0.9324

  
🩺 Clinical Takeaways

  

  
Perfect Sensitivity on Test Cohort: Hist Gradient Boosting achieved a Recall of 1.0000 (100%), meaning zero malignant tumors were missed (0 False Negatives) on the unseen test split.

  
Ensemble Superiority: Every ensemble model comfortably outperformed the baseline Decision Tree across all metrics.

  
Generalization Stability: Both bagging (Random Forest, Extra Trees) and boosting (Hist Gradient Boosting) maintained high ROC-AUC scores (>0.99), proving robust discriminative thresholds.

  

  

  
📁 Repository Structure

  
text
├── breast_cancer_diagnostic_benchmark.ipynb   # Main interactive notebook (EDA + Benchmark)
├── requirements.txt                           # Project dependencies
└── README.md                                  # Project documentation

  

  
🚀 Quickstart Guide

  
1. Clone the Repository

  
bash
git clone https://github.com/<your-username>/breast-cancer-diagnostic-benchmark.git
cd breast-cancer-diagnostic-benchmark

  
2. Set Up Environment & Install Dependencies

  
bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
pip install -r requirements.txt

  
requirements.txt

  
text
numpy>=1.22.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0

  
3. Launch Notebook

  
bash
jupyter notebook breast_cancer_diagnostic_benchmark.ipynb
