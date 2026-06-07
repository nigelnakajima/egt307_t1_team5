# egt307_t1_team5
## Team members:
* Nigel
* Yihao
* Jayden

## Contributions:
* cleaning.py - Nigel
* config.py- Nigel
* ingestion.py-Nigel
* preprocessing.py - Yihao
* synthetic_data_generation.py - Jayden
* training.py - Yihao

## How to run the pipeline
1. Ensure you have Docker installed and running.
2. Navigate to project folder
3. To build the pipeline, run: docker compose build pipeline
4. To run it run: docker compose up pipeline

## How to start development environment
1. Ensure you have Docker installed and running.
2. Navigate to project folder
3. To build the pipeline, run: docker compose build jupyter
4. To run it run: docker compose up jupyter

## Summary of EDA key findings
There are missing data in 4 columns:Humidity, MetalOxideSensor_Unit2, CO_GasSensor, and Ambient Light Level. These missing values appear to be missing completely at random.

There are inconsistent naming in 2 columns:HVAC Operation Mode and Activity Level.

There exists temperature values that are extremely high(280-310) and are assumed to be in Kelvin. These values appear at random.

There are also humidity values below 0 and above 100, which is impossible as humidity is recorded in percentage and appear completely at random.

Numerical data are highly correlated except for temperature and humidity, which after further analysis was found to have a non-linear relationship. This suggests that linear machine learning models may not be effective in this scenario.

HVAC Operation Mode and Ambient Light Level have little to no impact on the rest of the data and is not useful in this scenario.

Dimensionality reduction could be an effective method if the input data becomes much bigger in size, but is unnecessary in this scenario and would have the drawback of reducing the explainability of the model.

Lastly, training a random forest model revealed that the categorical data had little to no importance to predicting activity level and can be dropped without any drawback.

## Explain and justify features that are engineered
For feature engineering, the first step we did was to use KNN to impute missing data which ensures that we do not lose any valuable information. To ensure that imputation was effective, we compared the before and after information matrix to ensure that information was not lost.

Secondly, we dropped the categorical data as we found that they did not contribute much to predicting activity level as mentioned in our EDA. This proved effective as it did not change the metrics of our selected models.

Lastly, we generated synthetic data using SMOTE which works by finding k-nearest neighbours and interpolating data that fits between them. This was effective as activity level had an unbalanced number of classes and this helped to create more data for the minority classes.

## Explanation of choice of models and justify any tuning methods used
We used Random Forest, XGBoost, and SVM because they are popular machine learning models that performs well on classification tasks and handle different types of data.

Before training, numerical data was scaled using StandardScaler and categorical data was converted using One-Hot Encoding in order to help the models to learn more effectively.

To improve the performance of the models, RandomizedSearchCV was used to test different hyperparameter combinations to find the best settings for each model. Stratified Cross Validation was used to ensure models are trained and evaluated fairly as each subset of the data maintains the same proportion of class labels as the original dataset. For XGBoost, early stopping was applied to reduce overfitting. Class weights and synthetic data generation were used to help the models learn from underrepresented classes.


## Explain any specific choice of metrics that are important to the problem statement
The models are evaluated using Accuracy, Precision, Recall, and F1-Score.

1. Accuracy measures the overall percentage of correct activity level predictions.
2. Precision measures how often the predicted activity level is correct.
3. Recall measures how well the model identifies the actual activity level of residents.
4. F1-Score combines Precision and Recall into a single score.

The weighted F1-Score was selected as the main metric for model tuning because the dataset may contain imbalanced activity levels. A high F1-Score indicates that the model can accurately identify activity patterns while minimising incorrect predictions. For the problem statement, incorrect classifications could reduce the effectiveness of the early warning system used to monitor elderly residents and detect potential health concerns.
