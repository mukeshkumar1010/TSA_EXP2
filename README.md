## Devloped by: Mukesh Kumar S
## Register Number: 212223240099
##  Date: 2-03-2025

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:

1. Import necessary libraries (NumPy, Matplotlib)

2. Load the dataset

3. Calculate the linear trend values using least square method

4. Calculate the polynomial trend values using least square method

5. Visualise the results

### PROGRAM:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('/content/AirPassengers.csv',parse_dates=['Month'],index_col='Month')

data.head()

resampled_data = data['#Passengers'].resample('Y').sum().to_frame()
resampled_data.head()

resampled_data.index = resampled_data.index.year

resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Month': 'Year'}, inplace=True)

resampled_data.head()

years = resampled_data['Year'].tolist()
passengers = resampled_data['#Passengers'].tolist()

```
linear trend estimation
```py
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, passengers)]

n = len(years)
b = (n * sum(xy) - sum(passengers) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(passengers) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

```

Polynomial Trend Estimation (Degree 2)

```py
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, passengers)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(passengers), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

```

Visualising results

```py
print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

resampled_data.set_index('Year',inplace=True)

resampled_data['#Passengers'].plot(kind='line',color='blue',marker='o') #alpha=0.3 makes em transparent
resampled_data['Linear Trend'].plot(kind='line',color='black',linestyle='--')

resampled_data['#Passengers'].plot(kind='line',color='blue',marker='o')
resampled_data['Polynomial Trend'].plot(kind='line',color='black',marker='o')

```

### OUTPUT

Trend Equations:

![image](https://github.com/user-attachments/assets/9c887dc7-cf3a-4f54-9402-50690ddab2ff)


Linear Trend Estimation plot

![image](https://github.com/user-attachments/assets/11b5dbaf-8392-4b40-ac1d-87076ca5e1a9)


Polynomial Trend Estimation (Degree 2) plot

![image](https://github.com/user-attachments/assets/fbce8816-27fa-41b0-b1c2-4dab719c733f)



### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
