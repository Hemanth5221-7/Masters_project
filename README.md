# SMS Spam Detection Using Machine Learning

## Overview
This project develops an SMS spam detection system using machine learning and natural language processing. The aim is to classify SMS messages as either **ham** (legitimate) or **spam** (unwanted/malicious). The project compares three models: **Random Forest**, **Support Vector Classification (SVC)**, and **XGBoost**.

## Dataset
The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

- Source: UCI Machine Learning Repository
- Dataset Link: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains 5,574 labelled SMS messages:
- Ham: 4,825
- Spam: 747

## Project Workflow
1. Load and inspect the dataset
2. Preprocess SMS text
3. Encode labels
4. Split data into training and testing sets
5. Convert text into TF-IDF features
6. Handle class imbalance using SMOTE
7. Train baseline models
8. Tune models using GridSearchCV
9. Evaluate performance using multiple metrics

## Preprocessing
The following preprocessing steps were applied:
- Lowercasing
- Removal of digits
- Removal of punctuation
- Tokenisation
- Stopword removal
- Lemmatisation

## Feature Extraction
Text was transformed using:

python
TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

Requirements
Install the main dependencies with:

pip install pandas numpy matplotlib seaborn scikit-learn nltk imbalanced-learn xgboost
