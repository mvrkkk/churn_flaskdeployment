import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)
clf = joblib.load('./model/model.pkl')
dtypes = joblib.load('./model/dtypes.pkl')
cols_order = joblib.load('./model/cols_order.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    
    json_ = request.json
    df_input = pd.DataFrame.from_dict(json_, orient='index')
    df_input = df_input[cols_order]
    df_input = df_input.astype(dtypes)
    df_input = df_input.replace(" ", np.nan)
    
    
    predictions = clf.predict_proba(df_input)[:, 1]
    idx = df_input.index
    df_output = {'index': idx, 'probability_to_churn': predictions}
    
    
    return jsonify(pd.DataFrame(df_output).to_dict(orient='records'))
    
if __name__ == '__main__':
    app.run(port=8080, debug=True)