# CMI Gesture Detection using Sensor Data

## Overview
This project is based on the Kaggle competition **CMI - Detect Behavior with Sensor Data**. The notebook performs complete machine learning preprocessing and classification on sensor data to predict human gestures.

The workflow includes data loading, missing value handling, label encoding, outlier analysis, outlier treatment, model training, evaluation, and model saving for future predictions.

## Features

- Dataset download using KaggleHub
- Missing value detection and handling
- Categorical feature encoding with LabelEncoder
- Outlier detection using the IQR method
- Outlier visualization
- Outlier treatment
- Exploratory data analysis
- ExtraTreesClassifier model training
- Model evaluation using Accuracy Score
- Model export using Joblib

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- KaggleHub

## Machine Learning Pipeline

1. Data Loading
2. Missing Value Handling
3. Label Encoding
4. Outlier Detection
5. Outlier Treatment
6. Model Training
7. Model Evaluation
8. Model Saving

## Model

- ExtraTreesClassifier
- random_state = 42
- n_jobs = -1

## Future Improvements

- LightGBM
- XGBoost
- Hyperparameter Tuning
- Streamlit Deployment

## Author

Prince Yadav

BCA Student | Data Science Learner
