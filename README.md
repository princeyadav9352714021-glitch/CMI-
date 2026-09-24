# CMI: Detect Behavior with Sensor Data

## Project Overview

This project addresses the Kaggle **CMI - Detect Behavior with Sensor Data** classification problem. The objective is to identify the gesture performed in each sensor sequence using data collected from wearable devices.

The dataset includes:

- Accelerometer measurements (`acc_*`) for linear movement
- Rotation measurements (`rot_*`) for orientation and angular movement
- Thermal measurements (`thm_*`)
- Time-of-flight measurements (`tof_*`) for distance and proximity

The target column is `gesture`, which contains 18 gesture classes. The training dataset has approximately 574,945 rows and 341 columns. Several rows belong to the same `sequence_id`, so the data represents time-series sensor activity rather than independent records.

## Approach and Methods

The project follows an end-to-end machine learning workflow:

1. Authenticate with Kaggle and download the competition dataset using `kagglehub`.
2. Load and inspect the training data with pandas.
3. Check column types, class counts, unique values, and missing values.
4. Handle missing categorical values with the mode and missing numerical values with the median.
5. Convert categorical values into numerical values using label encoding.
6. Detect extreme numerical values using the interquartile range (IQR) method.
7. Cap extreme values at the IQR limits instead of deleting rows, preserving the available sensor sequences.
8. Separate the sensor features from identifiers and the target column.
9. Split the data into training and validation sets using an 80/20 stratified split.
10. Train and evaluate a multiclass classification model.

The Streamlit application also provides a sequence-level feature engineering workflow. It groups the raw records by `sequence_id` and calculates the mean, standard deviation, minimum, and maximum for each sensor feature. These statistical summaries describe the movement in a complete sequence and produce one feature vector per sequence.

## Algorithm Used

The model used is an **Extra Trees Classifier** (`ExtraTreesClassifier`). Extra Trees is an ensemble learning algorithm that builds many randomized decision trees and combines their predictions to classify the input.

The algorithm was appropriate for this problem because it:

- Handles a large number of numerical sensor features
- Learns nonlinear relationships between sensor signals and gestures
- Captures interactions between accelerometer, rotation, thermal, and ToF measurements
- Does not require feature scaling in the same way as distance-based models
- Reduces overfitting compared with using a single decision tree
- Performs well on structured tabular data

The notebook used 50 trees. The application uses 150 trees with `random_state=42` so that training is reproducible and the ensemble is more stable.

## Model Performance

The notebook achieved **81.84% validation accuracy** with the Extra Trees Classifier. In practical terms, the model correctly predicted the gesture for approximately 82 out of every 100 validation examples in that experiment.

The accuracy was calculated with:

```python
accuracy_score(y_test, model.predict(X_test))
```

The Streamlit application calculates and displays its validation accuracy dynamically after training. Its value may differ from the notebook result because the application uses sequence-level summaries and a different number of trees.

## Application Workflow

The accompanying Streamlit application allows users to:

- Download the CMI competition dataset through Kaggle
- Upload a local training or test CSV file
- Inspect the dataset and sensor columns
- Generate sequence-level sensor features
- Train the Extra Trees model
- View validation accuracy
- Compare actual and predicted gesture labels
- Confirm that `test.csv` follows the expected data structure

## Important Limitation

The reported 81.84% accuracy came from a row-level random split. Because multiple rows from one `sequence_id` can appear in both the training and validation sets, the model may see very similar sensor observations in both sets. This can lead to information leakage and may make the score higher than the model's performance on completely unseen sequences.

A stronger evaluation would split the data by `sequence_id`, ensuring that a complete sequence appears in only one split. To measure performance on new people, the split could instead be grouped by `subject`. Sequence-aware interpolation and explicit missing-value indicators for ToF sensors would also provide more reliable preprocessing than applying one global median to every numerical column.

## Conclusion

This project demonstrates a complete sensor-based gesture recognition pipeline. It combines data acquisition, cleaning, exploratory analysis, feature engineering, ensemble classification, validation, model persistence, and an interactive Streamlit interface. The Extra Trees model achieved a reported validation accuracy of **81.84%**, providing a strong baseline for recognizing gestures from wearable sensor data while leaving room for more rigorous sequence-based evaluation and improved time-series preprocessing.
