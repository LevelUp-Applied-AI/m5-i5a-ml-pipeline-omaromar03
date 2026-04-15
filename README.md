#  ML Evaluation Pipeline — Telecom Churn Prediction

##  Project Overview
This project implements a **reproducible machine learning evaluation pipeline** to predict customer churn using the Petra Telecom dataset.

The focus is not just on building models, but on **evaluating them correctly in an imbalanced classification setting**, where traditional metrics like accuracy can be misleading.

---

##  Problem Context
Customer churn prediction is a **high-impact business problem**:

- Missing a churner (false negative) = lost revenue
- Flagging a non-churner (false positive) = small operational cost

 Therefore, **Recall and F1 Score** are more important than Accuracy.

---

##  Technical Approach

### 1. Data Preparation
- Stratified train/test split (80/20)
- Target variable: `churned`

### 2. Preprocessing Pipeline
Built using `ColumnTransformer`:

- **Numeric features**
  - Median imputation
  - Standard scaling

- **Categorical features**
  - Most frequent imputation
  - One-hot encoding (`drop="first"`)

---

### 3. Models Evaluated
We compared five configurations:

| Model | Description |
|------|------------|
| Logistic Regression (default) | Baseline linear model |
| Logistic Regression (L1) | Feature selection via regularization |
| Ridge Classifier | L2 regularization |
| Most-Frequent Dummy | Predicts majority class |
| Stratified Dummy | Random prediction based on class distribution |

---

### 4. Evaluation Strategy
- 5-fold **Stratified Cross-Validation**
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - **F1 Score (primary metric)**

---

##  Key Results

### Cross-Validation Insights
- **Most-frequent Dummy** achieved high accuracy (~0.84) but:
-  Fails completely (F1 = 0.000)
- **Stratified Dummy** represents random guessing baseline
- Real models outperform both dummies on F1

👉 This confirms:
> Accuracy is misleading for imbalanced datasets.

---

##  Best Model

**Logistic Regression (L1, C=0.1)**

### Why?
- Highest F1 score among real models
- Strong recall (~0.63) → captures churners effectively
- Implicit feature selection via L1 regularization

---

##  Final Test Performance

| Metric | Value |
|-------|------|
| Accuracy | 0.650 |
| Precision | 0.267 |
| Recall | 0.653 |
| F1 Score | 0.379 |

---

##  Business Recommendation

I recommend **LogReg (L1, C=0.1)** because it provides the best balance between identifying churners and controlling false positives. While accuracy may suggest that simpler models perform better, the Most-frequent Dummy demonstrates that high accuracy can be achieved without learning anything useful. The selected model prioritizes recall, ensuring that a large portion of churners are detected, which is critical in retention strategies. Although precision is relatively low, this trade-off is acceptable in churn prediction. Compared to the Stratified Dummy, the model shows meaningful improvement, indicating real learning beyond random guessing. The test-set performance slightly exceeds cross-validation estimates, suggesting good generalization.

---

##  Limitations
- Linear models may not capture complex feature interactions
- Low precision indicates many false positives
- Feature engineering is minimal

---

##  Future Work
- Add tree-based models (Decision Trees, Random Forests)
- Perform hyperparameter tuning
- Improve feature engineering (interaction terms)
- Explore class imbalance techniques (SMOTE, threshold tuning)

---

##  Tech Stack
- Python
- pandas, numpy
- scikit-learn

---

##  How to Run

```bash
pip install -r requirements.txt
python evaluation_pipeline.py