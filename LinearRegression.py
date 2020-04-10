# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:08:47 2020

@author: deniz
"""
'''
---LOSS---
When we think about how we can assign a slope and intercept to fit a set of points, we have to define what the best fit is.
For each data point, we calculate loss, a number that measures how bad the model’s (in this case, the line’s) prediction was. You may have seen this being referred to as error.
We can think about loss as the squared distance from the point to the line. We do the squared distance (instead of just the distance) so that points above and below the line both contribute to total loss in the same way:
'''
'''
Tahmin edilen değer ile gerçek veri arasındaki farka loss deriz.Doğru bir prediction da
 loss çok az bir fark olması gerekir
'''
'''
Minimizing Loss
The goal of a linear regression model is to find the slope and intercept pair that minimizes loss on average across all of the data.
'''
import matplotlib.pyplot as plt
import numpy as np
x=[3,6,9]
y=[2,5,8]
m1=2
b1=4
m2=1
b2=1
plt.plot(x,y)
predict_value1=[x*m1+b1 for x in x]
plt.plot(predict_value1,y)
plt.plot(y,x)
predict_value2=[y*m2+b2 for y in y]
plt.plot(predict_value2,y)
loss1=0
for i in range(len(x)):
    loss1+=(x[i]-predict_value1[i])**2
loss2=0
for i in range(len(x)):
    loss2+=(y[i]-predict_value2[i])**2
print(loss1,loss2)
a=[5,10,15,20,25]#y=ax+b
x1=0.7
b1=5
pred_val=[a*x1+b1 for a in a]
plt.plot(a,pred_val)
'''
İster gerçek hayatta bir problemle uğraşalım ister bir yazılım ürünü ile, optimizasyon daima asıl hedeftir.
Optimizasyon temelde hedef alınan problem için en uygun sonucu (çıktıyı) elde etmek anlamına gelir.
Makine öğrenmesinde optimizasyon biraz daha farklıdır. Genel olarak, optimizasyon yaparken, 
verilerimizin nasıl göründüğünü ve iyileştirmek istediğimiz alanları tam olarak biliriz.
 Fakat makine öğrenmesinde, “yeni verilerimiz”in nasıl göründüğüne dair fikrimiz yoktur,
 tek başlarına optimize etmeye çalışırız. 
 Gradient Descent Kullanmamızdaki amac lossu en düşüğe indiren slope ve intercept ı bulmak.
'''
#Gradient Distance At B :-2/5*(y-(x*m+b))
#Gradient Distance At M :-2/5*(x*(y-(x*m+b))) Yani : x*B
def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff  
  return b_gradient

def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff  
  return m_gradient

#Your step_gradient function here
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]

def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
    return [b, m]
  
plt.show()
a=np.arange(9)
a=a.reshape(-1,1)
b=[12,18,26,30,25,32,45,50,52]
plt.scatter(a,b)
from sklearn.linear_model import LinearRegression
le=LinearRegression()
le.fit(a,b)
prediction = le.predict(a)
plt.plot(a,prediction)