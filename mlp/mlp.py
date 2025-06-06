import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import os

class MLP(nn.Module):
    def __init__(self, in_features=1, out_features=1, hidden_layers=[50, 30], activation='relu', batch_norm=False, dropout=0.1):
        super().__init__()
        layers = []

        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = h

        layers.append(nn.Linear(in_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'dataset', 'train_data.csv')
TEST_PATH = os.path.join(BASE_DIR, 'dataset', 'test_data.csv')
DEV_PATH = os.path.join(BASE_DIR, 'dataset', 'dev_data.csv')
SCALERX_PATH = os.path.join(BASE_DIR, 'dataset', 'scaler_x.pkl')
SCALERY_PATH = os.path.join(BASE_DIR, 'dataset', 'scaler_y.pkl')

#read csv
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_dev = pd.read_csv(DEV_PATH)

#load data from dataframe
X_train = df_train['x'].values.reshape(-1, 1)
y_train = df_train['y'].values.reshape(-1, 1)

X_test = df_test['x'].values.reshape(-1, 1)
y_test = df_test['y'].values.reshape(-1, 1)

X_dev = df_dev['x'].values.reshape(-1, 1)
y_dev = df_dev['y'].values.reshape(-1, 1)

#read scaler
scaler_x = joblib.load(SCALERX_PATH)
scaler_y = joblib.load(SCALERY_PATH)

test_x = scaler_x.inverse_transform(X_test)
test_y = scaler_y.inverse_transform(y_test)

def mlp_train(model, lr=0.005, epochs=300, scheduler_type=None, patience=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_dev_t = torch.tensor(X_dev, dtype=torch.float32).to(device)
    y_dev_t = torch.tensor(y_dev, dtype=torch.float32).to(device)

    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
    else:
        scheduler = None

    best_loss = float('inf')
    best_state = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_loss = criterion(model(X_dev_t), y_dev_t).item()

        if scheduler_type == 'step':
            scheduler.step()
        elif scheduler_type == 'plateau':
            scheduler.step(dev_loss)

        if dev_loss < best_loss:
            best_loss = dev_loss
            best_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if patience and no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 50 == 0 or epoch == epochs - 1:
            if scheduler_type == 'plateau':
                lr_info = f", LR={optimizer.param_groups[0]['lr']:.6f}"
            elif scheduler_type == 'step':
                lr_info = f", LR={scheduler.get_last_lr()[0]:.6f}"
            else:
                lr_info = ""
            print(f"Epoch {epoch}: Train Loss={loss.item():.6f}, Dev Loss={dev_loss:.6f}{lr_info}")

    model.load_state_dict(best_state)
    model.eval()
    return model

def mlp_predict(model, X):
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_t).cpu().numpy()
    return y_pred

#Basic MLP
print("Training Basic MLP model...")
mlp_basic = mlp_train(
    MLP(hidden_layers=[100, 50, 30], dropout=0.2, batch_norm=False, activation='relu'),
    lr=0.001,
    epochs=700,
    patience=None,
    scheduler_type=None
)

y_pred_basic = mlp_predict(mlp_basic, X_test)
mse_basic = mean_squared_error(test_y, scaler_y.inverse_transform(y_pred_basic))
print(f"Basic MLP Test MSE: {mse_basic:.6f}")

plt.figure(figsize=(10,6))
plt.scatter(test_x, test_y, label='True Test', s=15)
plt.scatter(test_x, scaler_y.inverse_transform(y_pred_basic), label='MLP Basic', s=15)
plt.title("MLP Basic Predictions on Test Set")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

#MLP Batch Norm + LeakyReLU
print("Training MLP model with Batch Norm and Leaky ReLU...")
mlp_bn_leaky = mlp_train(
    MLP(hidden_layers=[100, 50, 30],dropout=0.2, batch_norm=True, activation='leaky_relu'),
    lr=0.005,
    epochs=700
)
y_pred_bn_leaky = mlp_predict(mlp_bn_leaky, X_test)
mse_bn_leaky = mean_squared_error(test_y, scaler_y.inverse_transform(y_pred_bn_leaky))
print(f"MLP Batch Norm + LeakyReLU Test MSE: {mse_bn_leaky:.6f}")

plt.figure(figsize=(10,6))
plt.scatter(test_x, test_y, label='True Test', s=15)
plt.scatter(test_x, scaler_y.inverse_transform(y_pred_bn_leaky), label='MLP BN + LeakyReLU', s=15)
plt.title("MLP Batch Norm + LeakyReLU Predictions on Test Set")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

#MLP Early Stopping
print("Training MLP with Early Stopping...")
mlp_early_stopping = mlp_train(
    MLP(hidden_layers=[100, 50, 30], dropout=0.2, batch_norm=True, activation='leaky_relu'),
    lr=0.001,
    epochs=2000,
    patience=100,
)

y_pred_early_stopping = mlp_predict(mlp_early_stopping, X_test)
mse_early_stopping = mean_squared_error(test_y, scaler_y.inverse_transform(y_pred_early_stopping))
print(f"MLP Early Stopping Test MSE: {mse_early_stopping:.6f}")

plt.figure(figsize=(10,6))
plt.scatter(test_x, test_y, label='True Test', s=15)
plt.scatter(test_x, scaler_y.inverse_transform(y_pred_early_stopping), label='MLP Early Stopping', s=15)
plt.title("MLP Early Stopping Predictions on Test Set")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

#MLP LR Scheduler
print("Training MLP with LR Scheduler...")
mlp_lr_scheduler = mlp_train(
    MLP(hidden_layers=[100, 50, 30], dropout=0.2, batch_norm=True, activation='gelu'),
    lr=0.01,
    epochs=800,
    scheduler_type='plateau',
)

y_pred_lr_scheduler = mlp_predict(mlp_lr_scheduler, X_test)
mse_lr_scheduler = mean_squared_error(test_y, scaler_y.inverse_transform(y_pred_lr_scheduler))
print(f"MLP LR Scheduler Test MSE: {mse_lr_scheduler:.6f}")

plt.figure(figsize=(10,6))
plt.scatter(test_x, test_y, label='True Test', s=15)
plt.scatter(test_x, scaler_y.inverse_transform(y_pred_lr_scheduler), label='MLP LR Scheduler', s=15)
plt.title("MLP LR Scheduler Predictions on Test Set")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
