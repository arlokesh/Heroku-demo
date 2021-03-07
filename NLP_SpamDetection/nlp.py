# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


@author Lokesh AR
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


df= pd.read_csv("C:\\Users\\AG07256\\Downloads\\spam1.csv",encoding="latin-1")
#df.drop(['Unnamed:2','Unnamed : 3','Unnamed:4'],axis=1,inplace=True)

#Features and labels

df['label']=df['class'].map({'ham':0,'spam':1})
X=df['message']
y=df['label']

#extract feature with countVectorizer

cv=CountVectorizer()
X=cv.fit_transform(X)

pickle.dump(cv,open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
#Naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
message=np.array([['Email']])
print(clf.predict(message))
filename= 'nlp_model.pkl'
pickle.dump(clf,open(filename,'wb'))
