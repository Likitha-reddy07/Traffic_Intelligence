from flask import Flask, request, render_template,jsonify
import pickle
import numpy as np
import joblib
import streamlit as st
import matplotlib
import time
import os
import pandas 

app = Flask(_name_)

model=pickle.load(open('model.pkl','rb'))
scale=pickle.load(open('scaler.pkl','rb'))
encoder=pickle.load(open('encoder.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    input_feature=[float(x) for x in request.form.values() ]
    features_values=[np.array(input_feature)]
    names=[['holiday','temp','rain','snow','weather','year','month','day','hours','minutes','seconds']]
    data=pandas.DataFrame(features_values,columns=names)
    data=scale.fit_transform(data)
    data=pandas.DataFrame(data,columns=names)
    prediction=model.predict(data)
    print(prediction)
    text="Estimated Traffic Volume is : "
    return render_template("index.html",prediction_text=text+str(prediction))
if _name=="main_":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)