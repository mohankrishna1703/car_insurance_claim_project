🧠 Car Insurance Claim Prediction

🚗 Overview

This project aims to predict whether a customer will make an insurance claim based on their vehicle and policy details.
It involves the full data science workflow — including SQL integration, Exploratory Data Analysis (EDA), and Machine Learning to select the best predictive model.

🎯 Project Objectives

1. Analyze car insurance customer data.

2. Explore patterns and correlations using EDA.

3. Build and compare different machine learning models.

4. Select the best-performing model based on accuracy and F1/ROC-AUC scores.

5. Generate predictions for unseen data.

🗂️ Dataset Information

Files Used

| File                    | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| `train.csv`             | Training dataset with all features and target (`is_claim`) |
| `test.csv`              | Test dataset without target — used for predictions         |
| `sample_submission.csv` | Format for final predictions                               |
| `database.db`           | SQLite database used to store and fetch data using SQL     |

Key Columns in train.csv

| Column                | Description                                      |
| --------------------- | ------------------------------------------------ |
| `policy_id`           | Unique ID for each insurance policy              |
| `policy_tenure`       | Duration of policy coverage                      |
| `age_of_car`          | Age of the insured car                           |
| `age_of_policyholder` | Age of the customer                              |
| `segment`             | Type of car (A, B, C, etc.)                      |
| `fuel_type`           | Fuel type — Petrol/Diesel/Electric               |
| `ncap_rating`         | Car safety rating                                |
| `is_claim`            | **Target column** (1 = claim made, 0 = no claim) |

⚙️ Tech Stack

. Programming Language: Python

. Database: SQLite (via SQL queries in Python)

. Libraries:

     . pandas, numpy — data handling

     . matplotlib, seaborn — visualization

     . scikit-learn — ML modeling and metrics

     . imbalanced-learn — handling class imbalance (SMOTE)

     . sqlite3 — SQL integration
