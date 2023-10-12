from models import LSTM
from utils import get_device
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

device = get_device()


class PyTorchWrapper:
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, epochs=1000, lr=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.model = LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.numpy()

    def get_params(self, deep=True):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "epochs": self.epochs,
            "lr": self.lr
        }

    def set_params(self, **params):
        self.input_dim = params.get("input_dim", self.input_dim)
        self.hidden_dim = params.get("hidden_dim", self.hidden_dim)
        self.num_layers = params.get("num_layers", self.num_layers)
        self.output_dim = params.get("output_dim", self.output_dim)
        self.epochs = params.get("epochs", self.epochs)
        self.lr = params.get("lr", self.lr)
        # Update the model with new parameters
        self.model = LSTM(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim).to(device)
        return self  # Important: Ensure the updated instance is returned
