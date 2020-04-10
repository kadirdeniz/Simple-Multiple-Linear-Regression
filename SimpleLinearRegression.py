import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Linear Regression : y = mx+b
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
months=pd.DataFrame(months)
revenue=pd.DataFrame(revenue)

lr.fit(months,revenue)
y=lr.predict(months)
plt.plot(months,y)
plt.scatter(months,revenue)
print('Coef',lr.coef_,'İnterce',lr.intercept_)

print(lr.score(months,revenue)) 
    