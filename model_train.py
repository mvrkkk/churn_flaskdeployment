import pandas as pd
import numpy as np
import sys

from transformers import * 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline,make_pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.externals import joblib


pd.options.display.max_columns = None
pd.options.display.max_rows = 50
sys.path.append('./')


# імопртвання навчальної вибірки
df = pd.read_csv("./raw_data/churn_sample.csv")

cat_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
     'PhoneService', 'MultipleLines', 'InternetService',
     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod'
]
num_features = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

df.head()
df = df.replace(" ", np.nan)
X = df.drop(['customerID','Churn'],1)
y = df['Churn'].map({'Yes':1,'No':0})

pipe = Pipeline([
    ("features", FeatureUnion([
        ('categorical', make_pipeline(VariableSelector(names = cat_features), MultiColumnLabelEncoder())),
        ('numeric', make_pipeline(VariableSelector(names = num_features), SimpleImputer(), StandardScaler()))    
    ])
    )
]
)

X_t = pipe.fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_t ,y, test_size = .25 ,random_state = 111,stratify = y)
cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=111)
params = {
    'n_estimators':[100,250,500],
    'max_depth':[2,5,7,10,12],
    'max_features':['auto','sqrt'],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,5,10,12],
    'class_weight':['balanced',None]
}

grid = RandomizedSearchCV(RandomForestClassifier(), params, scoring='f1', cv=cv, verbose=1).fit(X_train, y_train)

pipe = Pipeline([
    ("features", FeatureUnion([
        ('categorical', make_pipeline(VariableSelector(names = cat_features), MultiColumnLabelEncoder())),
        ('numeric', make_pipeline(VariableSelector(names = num_features), SimpleImputer(), StandardScaler()))    
    ])
    ),
    ('prediction', grid.best_estimator_)
]
)

pipe.fit(X, y)
joblib.dump(pipe,'./model/model.pkl')