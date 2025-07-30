Space Titanic Machine Learning Project
This repository contains a contain Notebook (final_code.ipynb) that implements a machine learning pipeline to predict whether passengers on the Space Titanic were Transported to another dimension. The project covers data loading, extensive exploratory data analysis (EDA), robust preprocessing, and the implementation and evaluation of various machine learning models.

**Table of Contents**
Project Objectives
Dataset

Exploratory Data Analysis (EDA)

Data Preprocessing

Model Implementation and Evaluation

Model Performance Summary

Explainable AI (SHAP/LIME)

**Installation and Usage**

**Project Objective**
The primary goal of this project is to build a machine learning model that accurately predicts the Transported status of passengers aboard the Space Titanic. This involves:

Handling missing data.

Engineering relevant features.

Training and evaluating multiple classification models.

Comparing their performance to identify the best-suited model for this task.

**Dataset**

The project utilizes two main datasets:

train.csv: Training data with passenger information and the Transported target variable.


test.csv: Test data for making predictions, with 13 columns (lacking the Transported target).


sample_submission.csv: A sample submission file format.

**Exploratory Data Analysis (EDA)**

The EDA phase involved:

Initial Data Inspection 

Missing Value Analysis

Target Variable Distribution

Feature Distributions

Categorical Feature Analysis

Total Spending Analysis

Correlation Analysis

**Data Preprocessing**
A robust preprocessing pipeline was implemented to prepare the data:

**Data Splitting: **

The preprocessed data was split back into training and validation sets (X_train, X_test, y_train, y_test) for model training and evaluation. A test_size of 0.2 and random_state=42 were used, with stratify=y_original_train to maintain class distribution.

**Model Implementation and Evaluation**

Several machine learning models were implemented and evaluated for their performance on the validation set. Each model was trained with specific hyperparameters (tuned manually or based on common practices).

Models Used:
K-Nearest Neighbors (KNN)

AdaBoost Classifier

Random Forest Classifier

Decision Tree Classifier

Support Vector Machine (SVM)

Logistic Regression


**Evaluation Metrics:**
For each model, the following metrics were calculated on both the training and validation sets:

Accuracy
Precision
Recall
F1-Score
Classification Report
Confusion Matrix

**Model Performance Summary**
The performance of all trained models on the validation set is summarized below:

The Random Forest Classifier achieved the highest validation accuracy (0.7936) and a strong F1-score, indicating its robust performance.

**Installation and Usage**
To run this notebook, we'll need Python and the following libraries:

pip install pandas numpy scikit-learn matplotlib seaborn xgboost lime tensorflow keras

Steps to run the notebook:

**Clone this repository:**

git clone https://github.com/hertsGroup-1/Kaggle-Challenge

cd Kaggle-Challenge

Place your train.csv, test.csv, and sample_submission.csv files in the root directory of the cloned repository.

Open the final_code.ipynb notebook using google colab.

Run all cells in the notebook.

The notebook will:

Load and preprocess the data.

Train and evaluate the specified machine learning models.

Display performance metrics and confusion matrices.

Generate a submission.csv file with predictions on the test set.
