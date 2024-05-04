from flask import Flask, render_template,request
import joblib
# import pandas as pd
# import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
app=Flask(__name__)

# Load the pretrained model

lr=joblib.load('lr.pkl')

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
        
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure= int(request.form['BloodPressure'])
        Insulin= int(request.form['Insulin'])
        BMI= float(request.form['BMI'])
        Age= int(request.form['Age'])

    # Make prediction
        prediction = lr.predict([[Pregnancies,Glucose,BloodPressure,Insulin,BMI,Age]])[0]
        if prediction == 0:
            result = 'Not Diabetic'
        else:
            result = 'Diabetic'

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
