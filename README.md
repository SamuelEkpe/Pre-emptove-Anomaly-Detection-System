# Pre-emptove-Anomaly-Detection-System
A machine Learning Model to detect anomaly in equipment
 ## Predictive Maintenance Model

### Overview
This project aims to predict machine breakdowns by detecting anomalies in time-series data. The dataset consists of several features with one target column (y) representing anomalies (1: anomaly, 0: normal). The objective is to build a predictive maintenance solution that can preemptively identify equipment failures.

### Project Structure
AnomaData.csv: The dataset used for this project.
predictive_maintenance_model.pkl: The trained machine learning model saved for deployment.
Predictive_Maintenance_Report.pdf: A detailed report outlining design choices, performance evaluation, and future work.
pipeline.py: Python script containing the full code for the pipeline.
README.md: This file, explaining the project and how to run it.

### Steps to Reproduce
1. #### Clone the Repository
Download or clone the project folder to your local machine: git clone < your-repo-url>

2. #### Install Dependencies:
   Install the required python libraries using pip
   <pip install -r requirements.txt>
   The dependencies for this project include:
. pandas
. numpy
.scikit-learn
.matplotlib
.seaborn
.joblib

To install them manually, run:
< pip install pandas numpy scikit-learn joblib matplotlib seaborn>

### Running the Code
The main Python script (anoma_predictive_maintenance.ipynb) performs the following tasks:

Loads the dataset (AnomaData.csv).
Performs Exploratory Data Analysis (EDA).
Cleans the data and converts the time columns to the correct format.
Scales the features using StandardScaler.
Splits the dataset into training and testing sets.
Trains a Random Forest model.
Performs hyperparameter tuning using GridSearchCV.
Evaluates the model performance.
Saves the final model (predictive_maintenance_model.pkl) for future use.

run the pipepline:
< python anoma_predictive_maintenance.ipynb >

### Model Deployment
The final model is saved as predictive_maintenance_model.pkl. You can load it for future predictions using the following:

< import joblib>

< model = joblib.load('predictive_maintenance_model.pkl')>

< y_pred = model.predict(X_test)>

### Customizing the code

Model Training: If you want to try other models (e.g., Logistic Regression, Decision Trees), you can modify the section under "Model Selection."
Hyperparameter Tuning: Adjust the parameter grid in the GridSearchCV section to explore additional hyperparameters.

### Future Imporvement
. Class Imbalance: Implement techniques like SMOTE or downsampling to handle class imbalance better.
. Time-series Feature Engineering: Add advanced time-based features to improve prediction accuracy.
. Model Deployment: Integrate the model into a web or cloud-based platform for real-time predictions.

### Contact
If you encounter any issues or have further questions, feel free to reach out at [samuelgoldencyril@gmail.com].

