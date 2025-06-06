import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os

#Ground truth function
def f(x):
    return np.exp(-0.25*x) * np.cos(2*x)

#Generate data
def generate_data(num_points=1000, noise_std=0.05, x_range=(0, 4*np.pi)):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y_true = f(x)
    noise = np.random.normal(0, noise_std, size=num_points)
    y = y_true + noise
    return x.reshape(-1,1), y.reshape(-1,1), y_true.reshape(-1,1)

x_data, y_data, y_true = generate_data()

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x_data)
y_scaled = scaler_y.fit_transform(y_data)

#Divide datasets into train, dev, test sets
X_train_dev, X_test, y_train_dev, y_test = train_test_split(
    x_scaled, y_scaled, test_size=0.15, random_state=42)

X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_dev, y_train_dev, test_size=0.1765, random_state=42)

def inverse_transform(scaler_x, scaler_y, X, y):
    x_inv = scaler_x.inverse_transform(X)
    y_inv = scaler_y.inverse_transform(y)
    return x_inv.flatten(), y_inv.flatten()

#Inverse transform of scaled datasets
train_x, train_y = inverse_transform(scaler_x, scaler_y, X_train, y_train)
dev_x, dev_y = inverse_transform(scaler_x, scaler_y, X_dev, y_dev)
test_x, test_y = inverse_transform(scaler_x, scaler_y, X_test, y_test)

#Create DataFrames for each dataset
df_train = pd.DataFrame({'x': X_train.flatten(), 'y': y_train.flatten()})
df_dev = pd.DataFrame({'x': X_dev.flatten(), 'y': y_dev.flatten()})
df_test = pd.DataFrame({'x': X_test.flatten(), 'y': y_test.flatten()})

#Create path to datasets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'dataset', 'train_data.csv')
TEST_PATH = os.path.join(BASE_DIR, 'dataset', 'test_data.csv')
DEV_PATH = os.path.join(BASE_DIR, 'dataset', 'dev_data.csv')
SCALERX_PATH = os.path.join(BASE_DIR, 'dataset', 'scaler_x.pkl')
SCALERY_PATH = os.path.join(BASE_DIR, 'dataset', 'scaler_y.pkl')

#Write each DataFrame to a separate CSV file
df_train.to_csv(TRAIN_PATH, index=False)
df_dev.to_csv(TEST_PATH, index=False)
df_test.to_csv(DEV_PATH, index=False)

#Save scalers
joblib.dump(scaler_x, SCALERX_PATH)
joblib.dump(scaler_y, SCALERY_PATH)

if __name__ == "__main__":
    print(f'Train samples: {X_train.shape[0]}')
    print(f'Dev samples: {X_dev.shape[0]}')
    print(f'Test samples: {X_test.shape[0]}')

    plt.figure(figsize=(10,6))
    plt.scatter(X_train, y_train, label='Train', s=10, color='blue')
    plt.scatter(X_dev, y_dev, label='Dev', s=10, color='green')
    plt.scatter(X_test, y_test, label='Test', s=10, color='red')
    plt.title('Train / Dev / Test splits (Scaled Data)')
    plt.xlabel('x (scaled)')
    plt.ylabel('y (scaled)')
    plt.legend()
    plt.grid(True)
    plt.show()