# SUMDU bachelor diploma project
Bachelor diploma project - predicting telecom churn with deployment of model with Flask API

## Dependencies
```
flask
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
```

```pip install -r requirements.txt```

## Structure

Repository contains:
- model_train.py - training RandomForestClassifier
- app.py - running an application service

Folders 
- **model** contains pickled objects of pipeline, dtypes and columns order
- **notebook** contains EDA, model_train and API testing

## Running API
python app.py <port>

## Endpoints
**/predict (POST)**

Returns an array of predictions given a JSON object representing independent variables. Here's a sample input:
```
{2: {'gender': 'Male',
  'SeniorCitizen': '0',
  'Partner': 'No',
  'Dependents': 'No',
  'tenure': '2',
  'PhoneService': 'Yes',
  'MultipleLines': 'No',
  'InternetService': 'DSL',
  'OnlineSecurity': 'Yes',
  'OnlineBackup': 'Yes',
  'DeviceProtection': 'No',
  'TechSupport': 'No',
  'StreamingTV': 'No',
  'StreamingMovies': 'No',
  'Contract': 'Month-to-month',
  'PaperlessBilling': 'Yes',
  'PaymentMethod': 'Mailed check',
  'MonthlyCharges': '53.85',
  'TotalCharges': '108.15'}}
  ```
  
and sample output:

```[{'index': '2', 'probability_to_churn': 0.678650531321351}]```
