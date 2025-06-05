import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
from scipy.interpolate import CubicSpline

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

#if __name__ == "__main__":
sorted_indices = np.argsort(X_train.flatten())
X_train_sorted = X_train.flatten()[sorted_indices]
y_train_sorted = y_train.flatten()[sorted_indices]

cs = CubicSpline(X_train_sorted, y_train_sorted)

y_pred_spline = cs(X_test.flatten())

y_pred_spline_orig = scaler_y.inverse_transform(y_pred_spline.reshape(-1,1))

mse_spline = mean_squared_error(test_y, y_pred_spline_orig)
print(f"Spline Interpolation Test MSE: {mse_spline:.6f}")

plt.figure(figsize=(10,6))
plt.scatter(test_x, test_y, label='True Test Data', s=15)
plt.scatter(test_x, y_pred_spline_orig, label='Spline Prediction', s=15)
plt.title('Spline Interpolation Predictions on Test Set')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()