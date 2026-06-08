# Predicting Online Shopping Intent: A Comparative Study of New and Returning Visitors with Machine Learning Models

Master's Thesis Project | Tilburg University | MSc Data Science and Society

## Project Overview
This project predicts whether online shoppers will make a purchase based on their browsing behavior and session characteristics.

Unlike most previous studies that analyze all visitors as a single group, this project investigates new visitors and returning visitors separately to better understand behavioral differences and improve prediction performance.

Four machine learning models were evaluated, including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. Model interpretability was further explored using SHAP, while SMOTE was applied to address class imbalance.

This project was conducted as part of my MSc thesis in Data Science and Society at Tilburg University.

## Research Questions
1. How do different machine learning models (linear and non-linear) perform in predicting purchase intent for new and returning visitors?

2. Which features have the strongest impact on purchase intent for each group of customers?

3. What is the impact of handling class imbalance with SMOTE on the predictive performance of models for new and returning visitors?

## Dataset
**Online Shoppers Purchasing Intention Dataset** 
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- Records: 12,330 user sessions
- Features: 18 numerical and categorical variables
- Target Variable: Revenue (Purchase / No Purchase)
- Customer Segments:
  - New Visitors
  - Returning Visitors

## Technologies Used
Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), SHAP, Matplotlib)

