# House-Price-Prediction

# Objective:

To build and evaluate a Linear Regression model that predicts the median value of owner-occupied houses (MEDV) in Boston suburbs based on various housing and neighborhood features.

# Dataset Used:

Name: Boston Housing Dataset

Source: fetch_openml(name='boston') from scikit-learn

Attributes:

The dataset contains 506 rows and 14 columns.

13 are features (independent variables)

1 is the target (MEDV) - Median value of homes in $1000s

#  Tools and Libraries Used:

Python 3

Pandas – for data handling

Matplotlib / Seaborn – for visualization

Scikit-learn (sklearn) – for machine learning models

# Machine Learning Task:

Supervised Learning

Regression Problem

Model Used: LinearRegression() from sklearn.linear_model

# Workflow:

Load Dataset from OpenML

Explore Data (check features, target, and column types)

Preprocess Data: Ensure numeric types, handle missing values

Split Data into training and test sets (80/20)

Train Linear Regression Model using training data

Predict on test set

#  Learning Outcomes:

Understand how Linear Regression works in real-world datasets

Learn how to clean and prepare data for ML

Interpret model accuracy using evaluation metrics

Visualize the performance of a regression model


Evaluate Model using:

Mean Squared Error (MSE)

R² Score

Visualize Results using a scatter plot of actual vs predicted prices
