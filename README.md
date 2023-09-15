# Credit Card Fraud Detection Analysis Report
Fraud &amp; Anomaly Detection

![fraud](/oktoberfest2.jpeg)

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Class Distribution](#class-distribution)
  - [Feature Visualization](#feature-visualization)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
  - [Performance Metrics](#performance-metrics)
  - [Learning Curves](#learning-curves)
- [Results](#results)
- [Conclusion](#conclusion)
- [Clone](#clone)
- [Installation](#installation)

## Introduction

Credit card fraud is a significant concern in the financial industry, leading to substantial financial losses for both consumers and businesses. In this analysis, we employed various machine learning models and anomaly detection techniques to identify and mitigate credit card fraud.

## Dataset

The dataset used for this analysis is the "Credit Card Fraud Detection" dataset, which can be found on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Data Description

The dataset contains the following columns:

- Time
- V1, V2, V3, ... V28: Transformed features to protect sensitive information
- Amount: Transaction amount
- Class: 0 for "No Fraud," 1 for "Fraud"

Here's a sample of the dataset:

|    |   Time |       V1 |        V2 |       V3 |        V4 |       V5 |       V6 |       V7 |       V8 |        V9 | ... |      V21 |        V22 |        V23 |      V24 |        V25 |        V26 |        V27 |       V28 |   Amount |   Class |
|---:|-------:|---------:|----------:|---------:|----------:|---------:|---------:|---------:|---------:|----------:|---:|---------:|-----------:|-----------:|---------:|-----------:|-----------:|-----------:|----------:|---------:|--------:|
|  0 |      0 | -1.35981 | -0.072781 |  2.53635 |   1.37815 | -0.338321 |  0.462388 |  0.239599 |  0.098698 |  0.363787 | ... | -0.018307 |  0.277838  | -0.110474  |  0.066928 |  0.128539  | -0.189115  |  0.133558  | -0.021053 |  149.62 |       0 |
|  1 |      0 |  1.19186 |  0.266151 |  0.16648 |   0.448154 |  0.060018 | -0.082361 | -0.078803 |  0.085102 | -0.255425 | ... | -0.225775 | -0.638672  |  0.101288  | -0.339846 |  0.16717   |  0.125895  | -0.008983  |  0.014724 |    2.69 |       0 |
|  2 |      1 | -1.35835 | -1.34016  |  1.77321 |   0.37978  | -0.503198 |  1.800499 |  0.791461 |  0.247676 | -1.51465  | ... |  0.247998 |  0.771679  |  0.909412  | -0.689281 | -0.327642  | -0.139097  | -0.055353  | -0.059752 |  378.66 |       0 |
|  3 |      1 | -0.966272| -0.185226 |  1.79299 |  -0.863291 | -0.010309 |  1.247203 |  0.237609 |  0.377436 | -1.38702  | ... | -0.1083   |  0.00527396| -0.190321  | -1.175575 |  0.647376  | -0.221929  |  0.062723  |  0.061458 |  123.5  |       0 |
|  4 |      2 | -1.15823 |  0.877737 |  1.54872 |   0.403034 | -0.407193 |  0.095921 |  0.592941 | -0.270533 |  0.817739 | ... | -0.009431 |  0.798278  | -0.137458  |  0.141267 | -0.20601   |  0.502292  |  0.219422  |  0.215153 |   69.99 |       0 |

## Data Preprocessing

I began by preprocessing the dataset, which involved standardizing and transforming the features using techniques like Yeo-Johnson power transformation. This step ensured that the data was suitable for modeling.

## Exploratory Data Analysis

### Class Distribution

I explored the distribution of classes in the dataset, revealing a severe class imbalance between 'No Fraud' and 'Fraud' transactions. This imbalance had a significant impact on model performance.

### Feature Visualization

I visualized the distributions of transformed features, gaining insights into their characteristics and potential for discrimination between classes. Histograms with KDE overlays provided a clear overview of each feature's distribution.

## Model Building

I experimented with several machine learning models, including:

- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier
- Logistic Regression

Additionally, I utilized specialized anomaly detection techniques, such as Isolation Forest and Local Outlier Factor (LOF), to identify fraudulent transactions.

## Model Evaluation

### Performance Metrics

For classification models, I assessed their performance using metrics such as precision, recall, F1-score, ROC AUC score, and the confusion matrix. These metrics allowed us to gauge how well the models detected fraud while controlling false positives.

### Learning Curves

Learning curves provided insights into model behavior concerning training set size, helping us understand how performance scales with additional data.

## Results

- The Random Forest and XGBoost classifiers demonstrated strong performance in detecting fraudulent transactions, with AUC-ROC scores above 0.90.
- Logistic Regression, with class-weight adjustments, improved performance, achieving higher precision and recall.
- Isolation Forest and LOF showed promise in identifying anomalies, with ROC AUC scores above 0.90.

## Conclusion

In this analysis, I explored various machine learning models and anomaly detection techniques for credit card fraud detection. My findings indicate that ensemble methods like Random Forest and XGBoost, along with proper class-weight adjustments, offer effective solutions for detecting fraud. Additionally, specialized anomaly detection methods, such as Isolation Forest and LOF, can complement traditional classifiers.

However, further exploration is needed to address the class imbalance issue and fine-tune models for improved recall while maintaining high precision. Continuous monitoring and adaptation to emerging fraud patterns are crucial for robust fraud detection systems.

By combining the strengths of machine learning and anomaly detection techniques, financial institutions can enhance their ability to detect and prevent credit card fraud, safeguarding the interests of both customers and businesses.

## Clone

To clone this repository, use the following command:

```bash
git clone https://github.com/hossein-zandinejad/your-repository.git
```

## Installation

To install the required libraries for this project, you can use a package manager like pip. Run the following command:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install plotly
pip install scikit-learn
pip install scipy
pip install statsmodels
pip install xgboost
pip install imbalanced-learn
```
