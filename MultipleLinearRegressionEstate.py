"""
Created on Sat Apr  4 01:42:07 2020

@author: deniz
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

veriler = pd.read_csv('manhattan.csv')
veriler = veriler.iloc[:,1:-2]
veriler=veriler.head(250)
rent=pd.DataFrame(veriler['rent'])
df=pd.DataFrame(veriler.drop(['rent'],1))

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
rent=ss.fit_transform(rent)
df=ss.fit_transform(df)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test  =train_test_split(df,rent,train_size=0.8)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train,y_train)
print(lr.score(x_train,y_train))
print(lr.coef_)
print(lr.intercept_)

predict = lr.predict(x_test)

def loss(x,y):
    distance = 0
    sqrt_distance=0
    for i in range(len(x)):
       sqrt_distance += (x[i]-y[i])**2
    distance=sqrt_distance**0.5
    
    return distance


print('Loss :',loss(predict,y_test))