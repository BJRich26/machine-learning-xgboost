# machine-learning-xgboost ReadMe File


# Cardiovascular Disease Prediction Using XGBoost

## Project Overview
The goal of this project is to predict whether a patient experienced a cardiovascular disease event using demographic, clinical, and sleep-related health features from the Sleep Heart Health Study dataset.

The model in this workbook uses Extreme Gradient Boosting, also known as XGBoost, to perform binary classification on the target variable cvd_event.

## Research Objective

The main predictive question for this workbook is:

1. Can cardiovascular disease event occurrence be predicted using sleep metrics, demographic variables, and baseline health indicators?

The supporting questions are:

2. What are the most important features for predicting cardiovascular disease risk?

3.Do sleep-related features such as sleep apnea severity, oxygen desaturation, and sleep time contribute meaningful predictive value?


## Dataset Description

The model uses a prepared cardiovascular disease prediction dataset built from Sleep Heart Health Study data. The dataset includes demographic variables, traditional health indicators, sleep-study measurements, oxygen saturation variables, and the binary target variable cvd_event.

The target variable is cvd_event.

A value of 0 means the patient did not experience a cardiovascular disease event.

A value of 1 means the patient did experience a cardiovascular disease event.

The original modeling dataset had around 5,000 records. The target distribution was imbalanced, with most records belonging to the non-CVD class. Because of this imbalance, I used random undersampling to create a more balanced dataset for training and evaluation.

## File Description

The main file in this repository is XGBoost_Model.ipynb.

This notebook contains the full individual XGBoost modeling process, including project background, feature selection, sampling, model training, evaluation, SHAP interpretation, limitations, and future improvement opportunities.

Depending on the final repository setup, this project may also include the model-ready dataset, a saved model file, a Python script version of the notebook, and a requirements file.

## Notebook Workflow

The workbook follows this process:

1. Introduces the cardiovascular disease prediction problem and explains why it matters.

2. Loads the model-ready cardiovascular disease dataset.

3. Reviews the available variables and target variable.

4. Uses a correlation matrix and domain knowledge to select relevant features.

5. Reduces multicollinearity concerns by avoiding highly correlated predictors.

6. Selects a final group of demographic, health, sleep, and oxygen saturation features.

7. Checks the original target class distribution.

8. Applies random undersampling to reduce class imbalance.

9. Splits the balanced dataset into training and testing sets.

10. Trains an XGBoost classification model.

11. Evaluates the model using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

12. Uses built-in feature importance to understand which features the model relied on most.

13. Uses SHAP analysis to explain the most influential features.

14. Discusses the final model results, limitations, and possible future improvements.

## Feature Selection Summary

The model used correlation analysis and healthcare domain knowledge to select features. The purpose of this step was to reduce redundant variables, avoid strong multicollinearity, and keep predictors that had a reasonable connection to cardiovascular disease risk.

The final feature set included demographic variables, traditional cardiovascular health indicators, sleep-related variables, and oxygen saturation measurements.

Key feature groups included:

Demographic features such as age and gender.

Health indicators such as body mass index, waist circumference, systolic blood pressure, diastolic blood pressure, hypertension, diabetes, and smoking status.

Sleep-related indicators such as sleep time, sleep efficiency, sleep apnea severity, and obstructive apnea-hypopnea measures.

Oxygen saturation indicators such as percentage of sleep time below 90% oxygen saturation and non-REM oxygen saturation.

## Class Imbalance Handling

The original dataset was imbalanced, with approximately 76% of records belonging to the non-CVD class and approximately 24% belonging to the CVD class.

Because healthcare prediction should avoid missing positive disease cases, class imbalance was an important issue. To address this, I used random undersampling. The majority class was reduced to create an approximate 60/40 split between non-CVD and CVD cases.

This helped the model pay more attention to the minority CVD event class. However, undersampling also has a tradeoff because it removes a large amount of records that may contain useful information. So to compormise, I used a 60/40 instead of 50/50 to retain close to 3000 of the original 5000 records for anlaysis.

## Model Description

The model used in this workbook is XGBoost Classification.

XGBoost was selected because it works well with structured/tabular datasets and can capture nonlinear relationships between predictors. This is useful for cardiovascular prediction because disease risk may depend on complex interactions between age, blood pressure, diabetes, hypertension, sleep apnea, and oxygen saturation.

XGBoost also provides built-in feature importance and works with SHAP values, which makes the model easier to interpret even though it is more complex than a traditional logistic regression model due to its black box nature.

## Model Performance

The final XGBoost model produced the following performance results:

Accuracy: 72.2%

AUC-ROC: 76.7%

Recall: 59.0%

Precision: 67.4%

F1-Score: 62.95%

The confusion matrix showed:

291 true negatives

68 false positives

98 false negatives

141 true positives

The most concerning result is the 98 false negatives. In a healthcare setting, false negatives represent patients who actually had a cardiovascular disease event but were predicted as not having one. Because of this, recall is especially important for this type of project.

Although the XGBoost model had moderate predictive performance, it did not fully meet the project’s preferred goals of an AUC-ROC above 80% and a recall score around 70% to 75%. Because of this, the model was useful for comparison but was not selected as the final team model.

## SHAP Feature Importance Summary

SHAP analysis was used to better understand which variables had the strongest influence on the model’s predictions.

The most influential features were:

Age

Gender

Systolic blood pressure

Diastolic blood pressure

Hypertension

Sleep apnea severity

Oxygen desaturation

Sleep time

Diabetes

Waist circumference

The strongest predictors were traditional cardiovascular risk factors such as age, gender, blood pressure, hypertension, and diabetes. However, sleep-related variables such as sleep apnea severity, oxygen desaturation, and sleep time also appeared among the most important features.

This supports the project’s idea that sleep metrics can provide useful additional information when predicting cardiovascular disease risk.

## Key Findings

The XGBoost model showed that cardiovascular disease prediction is possible using a combination of demographic, clinical, and sleep-related variables.

Traditional health indicators remained the strongest predictors.

Sleep-related indicators still contributed meaningful predictive value.

The model achieved reasonable AUC performance but had a recall score that was too low for ideal healthcare screening use.

The model was helpful for feature comparison and explainability, but it was not chosen as the final team model because it missed too many actual CVD cases.

## Limitations

One limitation of this model is that XGBoost is more of a black-box model compared with logistic regression. SHAP values help improve interpretability, but the exact internal decision process is still harder to explain.

Another limitation is the recall score. The model missed 98 actual CVD cases of its testing batch, which would be concerning in a real medical screening environment since we could potential risk patients being delayed life saving treatment.

A third limitation is the use of random undersampling. While undersampling improved class balance, it also removed some non-CVD records that may have helped the model learn a more complete or complex patterns from the data.

The model also did not reach the preferred AUC-ROC and recall thresholds set by the project team.

Finally, the model was trained on observational study data, so the results show predictive relationships rather than causational relationships.

## Future Improvements

Future versions of this model could be improved by testing more sampling methods, such as SMOTE or class weighting.

The classification threshold could also be adjusted to improve recall and reduce false negatives.

More extensive hyperparameter tuning could be performed to improve model performance.

The model could be tested on an external dataset to evaluate generalizability.

Additional longitudinal health data could be used to better capture how cardiovascular risk changes over time.

The model could also be integrated into a Streamlit application for interactive cardiovascular risk prediction.

## How to Run This Workbook

To run this workbook, download or clone the GitHub repository to your local computer.

Make sure the dataset file is saved in the same folder as the notebook, unless the notebook file path is updated.

Open the XGBoost_Model.ipynb file using Jupyter Notebook, JupyterLab, Google Colab, or another notebook environment.

Install the required Python libraries before running the notebook.

Run the notebook cells from top to bottom.

Review the final output cells for the model performance metrics, confusion matrix, feature importance chart, and SHAP interpretation.

## Required Python Libraries

This workbook uses the following Python libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

xgboost

shap

These libraries should be included in the project’s requirements.txt file so the notebook can be reproduced.

## Suggested Repository Contents

The repository should include:

XGBoost_Model.ipynb

README.md

requirements.txt

model_ready_cvd_prediction.csv, if allowed for submission

A saved model file, such as a pkl or sav file, if required

A Python script version of the notebook, if required

## Project Context

This XGBoost notebook is one individual model contribution within a team cardiovascular disease prediction project.

The broader team project compared multiple machine learning models and selected the best-performing final model based on predictive performance, healthcare usefulness, and interpretability.

This workbook specifically documents the XGBoost model development process and explains why the model was useful for comparison, even though it was not selected as the final team model.

## Author

Brandon Richard

BSAN 6070: Introduction to Machine Learning

Spring 2026
