# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:32:06 2021

@author: AG07256
"""


from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib

filename='nlp_model.pkl'
clf=pickle.load(open(filename,'rb'))
cv=pickle.load(open('transform.pkl','rb'))
appnlp=Flask(__name__)

@appnlp.route('/')
def home():
    return render_template('home.html')

@appnlp.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result.html',prediction=my_prediction) 


if __name__=='__main__':
    appnlp.run(debug=True)         
