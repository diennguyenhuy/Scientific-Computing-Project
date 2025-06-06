import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
from scipy.interpolate import lagrange
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'dataset', 'train_data.csv')
TEST_PATH = os.path.join(BASE_DIR, 'dataset', 'test_data.csv')
SCALERX_PATH = os.path.join(BASE_DIR, 'dataset', 'scaler_x.pkl')
SCALERY_PATH = os.path.join(BASE_DIR, 'dataset', 'scaler_y.pkl')

#read csv
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

#load data from dataframe
X_train = df_train['x'].values.reshape(-1, 1)
y_train = df_train['y'].values.reshape(-1, 1)

X_test = df_test['x'].values.reshape(-1, 1)
y_test = df_test['y'].values.reshape(-1, 1)

#read scaler
scaler_x = joblib.load(SCALERX_PATH)
scaler_y = joblib.load(SCALERY_PATH)

test_x = scaler_x.inverse_transform(X_test)
test_y = scaler_y.inverse_transform(y_test)

def lagrange_interpolation(num_points_lagrange):
    indices = np.random.choice(len(X_train), num_points_lagrange, replace=False)
    X_train_lagrange = X_train[indices].flatten()
    y_train_lagrange = y_train[indices].flatten()

    poly_lagrange = lagrange(X_train_lagrange, y_train_lagrange)

    X_test_limited = np.clip(X_test.flatten(), np.min(X_train_lagrange), np.max(X_train_lagrange))

    y_pred_lagrange = poly_lagrange(X_test_limited)
    y_pred_lagrange = np.nan_to_num(y_pred_lagrange, nan=0.0, posinf=0.0, neginf=0.0)

    y_pred_lagrange_orig = scaler_y.inverse_transform(y_pred_lagrange.reshape(-1,1))

    mse_lagrange = mean_squared_error(test_y, y_pred_lagrange_orig)
    print(f"Lagrange Interpolation Test MSE ({num_points_lagrange} points): {mse_lagrange:.6f}")

    plt.figure(figsize=(10,6))
    plt.scatter(test_x, test_y, label='True Test')
    plt.scatter(test_x, y_pred_lagrange_orig, label=f'Lagrange Pred ({num_points_lagrange} pts)')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Lagrange Interpolation Predictions (based on {num_points_lagrange} points of Train set)')
    plt.show()

lagrange_interpolation(5)
lagrange_interpolation(10)
lagrange_interpolation(15)
lagrange_interpolation(20)
lagrange_interpolation(25)
