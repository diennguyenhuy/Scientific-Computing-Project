import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlockBN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.linear1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        return self.relu(out + x)

class ResidualMLPBN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_blocks=5, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlockBN(hidden_dim) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.dropout(x)
        return self.output_layer(x)
    
#read csv
df_train = pd.read_csv('scicompproject/dataset/train_data.csv')
df_test = pd.read_csv('scicompproject/dataset/test_data.csv')
df_dev = pd.read_csv('scicompproject/dataset/dev_data.csv')

#load data from dataframe
X_train = df_train['x'].values.reshape(-1, 1)
y_train = df_train['y'].values.reshape(-1, 1)

X_test = df_test['x'].values.reshape(-1, 1)
y_test = df_test['y'].values.reshape(-1, 1)

X_dev = df_dev['x'].values.reshape(-1, 1)
y_dev = df_dev['y'].values.reshape(-1, 1)

#read scaler
scaler_x = joblib.load('scicompproject/dataset/scaler_x.pkl')
scaler_y = joblib.load('scicompproject/dataset/scaler_y.pkl')

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
        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            dev_loss = criterion(model(X_dev_t), y_dev_t).item()

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
            lr_info = f", LR={scheduler.get_last_lr()[0]:.6f}" if scheduler else ""
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

if __name__ == "__main__":
    mlp_residual = mlp_train(
        ResidualMLPBN(),
        lr=0.001,
        epochs=700,
        patience=100
    )

    y_pred_residual = mlp_predict(mlp_residual, X_test)
    mse_residual = mean_squared_error(test_y, scaler_y.inverse_transform(y_pred_residual))
    print(f"Residual MLP Optimized Test MSE: {mse_residual:.6f}")

    plt.figure(figsize=(10,6))
    plt.scatter(test_x, test_y, label='True Test', s=15)
    plt.scatter(test_x, scaler_y.inverse_transform(y_pred_residual), label='Residual MLP Optimized', s=15)
    plt.title("Residual MLP Optimized Predictions on Test Set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()
