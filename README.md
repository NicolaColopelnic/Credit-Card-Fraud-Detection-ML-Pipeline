# Identifying Fraudulent Transactions with Feature Engineering and SMOTE

This project focuses on building a robust intelligent system capable of detecting fraudulent credit card transactions. Fraud detection is a classic imbalanced classification problem, where legitimate transactions vastly outnumber fraudulent ones.

The goal was to move beyond simple model training by implementing advanced feature engineering, handling class imbalance, and hyperparameter optimization to maximize the F1-score.

## Dataset Insights
Source: Kaggle Credit Card Fraud Dataset: https://www.kaggle.com/datasets/dhruvb2028/credit-card-fraud-dataset

Size: ~339,000 transactions.

Imbalance: 99.48% legitimate vs. 0.52% fraudulent.

## Key Technical Features
### 1. Feature Engineering - Standard data was transformed into high-signal predictors:

* Haversine Distance: Calculated the physical distance between the customer's home and the merchant location.

* Temporal Analysis: Extracted the hour of the transaction, identifying a high-risk window between 10 PM and 3 AM.

* Demographics: Calculated cardholder age from date of birth to identify age-related fraud patterns.

### 2. Solving Class Imbalance
* Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the training set. This prevented the model from being biased toward the majority class and improved the detection of rare fraud cases.

### 3. Model Comparison & Optimization
Evaluated multiple architectures using Precision, Recall, and F1-Score:

* Naive Bayes: Baseline probabilistic model.

* Decision Tree: Interpretable rules-based model.

* Random Forest: Ensemble method to reduce variance.

* Hyperparameter Tuning: Conducted a RandomizedSearchCV to optimize the decision tree, resulting in an improvement in F1-score over the baseline.

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | **F1-Score** |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Bayes** (Baseline) | 0.62 | 0.01 | 0.69 | 0.02 |
| **Decision Tree** (Initial) | 0.97 | 0.13 | **0.90** | 0.23 |
| **Random Forest** (Ensemble) | 0.99 | 0.27 | 0.88 | 0.41 |
| **Optimized Decision Tree** | **0.99** | **0.38** | 0.86 | **0.53** |

### Tools Used
* Python (Pandas, NumPy, Matplotlib, Seaborn)

* Scikit-Learn (machine learning, preprocessing, evaluation)

* Imbalanced-learn (SMOTE)



