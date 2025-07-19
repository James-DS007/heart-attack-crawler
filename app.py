# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 20:08:57 2025

@author: James
"""

from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')  # 加载模型

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从表单接收输入
        inputs = {
            'age': float(request.form['age']),
            'trestbps': float(request.form['trestbps']),
            'chol': float(request.form['chol']),
            'thalch': float(request.form['thalch']),
            'oldpeak': float(request.form['oldpeak']),
            'fbs': float(request.form['fbs']),
            'sex': request.form['sex'],
            'cp': request.form['cp'],
            'restecg': request.form['restecg'],
            'slope': request.form['slope'],
            'thal': request.form['thal'],
            'exang': request.form['exang'],
            'ca': request.form['ca']
        }
        
        # 构造 DataFrame
        input_df = pd.DataFrame([inputs])
        input_df['ca'] = input_df['ca'].astype(str)
        input_df['chol_trestbps_ratio'] = input_df['chol'] / (input_df['trestbps'] + 1)

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        result = f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'} (Risk score: {prob:.2f})"
        return render_template('index.html', result=result)
    
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
