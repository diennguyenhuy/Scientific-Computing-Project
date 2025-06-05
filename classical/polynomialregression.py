import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#read csv
df_train = pd.read_csv('scicompproject/dataset/train_data.csv')
df_test = pd.read_csv('scicompproject/dataset/test_data.csv')

#load data from dataframe
X_train = df_train['x'].values.reshape(-1, 1)
y_train = df_train['y'].values.reshape(-1, 1)

X_test = df_test['x'].values.reshape(-1, 1)
y_test = df_test['y'].values.reshape(-1, 1)

#read scaler
scaler_x = joblib.load('scicompproject/dataset/scaler_x.pkl')
scaler_y = joblib.load('scicompproject/dataset/scaler_y.pkl')

test_x = scaler_x.inverse_transform(X_test)
test_y = scaler_y.inverse_transform(y_test)

mse_poly = []

def polynomial_regression(degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    y_pred_poly = LinearRegression().fit(X_train_poly, y_train).predict(X_test_poly)

    y_pred_poly_orig = scaler_y.inverse_transform(y_pred_poly.reshape(-1,1))

    mse = mean_squared_error(test_y, y_pred_poly_orig)
    print(f"Polynomial Regression (degree {degree}) Test MSE: {mse:.6f}")
    mse_poly.append(mse)

    plt.figure(figsize=(8,6))
    plt.scatter(test_x, test_y, label='True Test Data', s=15)
    plt.scatter(test_x, y_pred_poly_orig, label=f'Poly Regression Prediction (degree {degree})', s=15)
    plt.title(f'Polynomial Regression (degree {degree}) Predictions on Test Set')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

for degree in range(1, 11):
    polynomial_regression(degree)

plt.figure(figsize=(8,6))
plt.plot(range(1, 11), mse_poly, marker='o', linestyle='-', color='b')
plt.title('MSE vs. Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.show()