import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(n_samples=1500):
    x = np.random.uniform(0, 10, n_samples)
    y = np.random.uniform(0, 10, n_samples)
    z = np.exp(np.sin(x) + np.cos(y)) + np.log(1 + x**2 + y**2)  
    return np.column_stack((x, y)), z

def prepare_tensors(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X_tensor, y_tensor

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(FeedForwardNN, self).__init__()
        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CascadeForwardNN(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(CascadeForwardNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for h in hidden_layers:
            self.hidden_layers.append(nn.Linear(prev_size + input_size, h))
            prev_size = h
        self.output_layer = nn.Linear(prev_size + input_size, 1)

    def forward(self, x):
        inputs = x
        for layer in self.hidden_layers:
            inputs = torch.cat((inputs, x), dim=1)
            inputs = torch.relu(layer(inputs))
        inputs = torch.cat((inputs, x), dim=1)
        return self.output_layer(inputs)

class ElmanNN(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(ElmanNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_states = []

        prev_size = input_size
        for h in hidden_layers:
            self.hidden_layers.append(nn.Linear(prev_size + h, h))
            self.hidden_states.append(torch.zeros(1, h))  
            prev_size = h

        self.output_layer = nn.Linear(prev_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]  
        inputs = x
        new_states = []
        for i, layer in enumerate(self.hidden_layers):
            hidden_expanded = self.hidden_states[i].expand(batch_size, -1)
            inputs = torch.cat((inputs, hidden_expanded), dim=1)
            inputs = torch.relu(layer(inputs))

            new_states.append(inputs.mean(dim=0, keepdim=True).detach())

        self.hidden_states = new_states  
        return self.output_layer(inputs)

def train_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    y_pred_test = model(X_test).detach().numpy()
    error = mean_absolute_error(y_test, y_pred_test) / np.mean(y_test.numpy())
    return error, loss_history

X, y = generate_data()
X_train, y_train = prepare_tensors(X[:1000], y[:1000])
X_test, y_test = prepare_tensors(X[1000:], y[1000:])

architectures = {
    "FeedForward": {
        "10 нейронів": ([10], FeedForwardNN(2, [10])),
        "20 нейронів": ([20], FeedForwardNN(2, [20]))
    },
    "CascadeForward": {
        "1 шар, 20 нейронів": ([20], CascadeForwardNN(2, [20])),
        "2 шари, 10 нейронів": ([10, 10], CascadeForwardNN(2, [10, 10]))
    },
    "Elman": {
        "1 шар, 15 нейронів": ([15], ElmanNN(2, [15])),
        "3 шари, 5 нейронів": ([5, 5, 5], ElmanNN(2, [5, 5, 5]))
    }
}

results = []
loss_histories = {}

for model_type, models in architectures.items():
    loss_histories[model_type] = {}
    for name, (layers, model) in models.items():
        error, loss_history = train_model(model, X_train, y_train, X_test, y_test)
        results.append({"Модель": f"{model_type} ({name})", "Середня відносна помилка": error})
        loss_histories[model_type][name] = loss_history

df_results = pd.DataFrame(results)
print(df_results)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x_vals = np.linspace(0, 10, 50)
y_vals = np.linspace(0, 10, 50)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z_grid = np.exp(np.sin(X_grid) + np.cos(Y_grid)) + np.log(1 + X_grid**2 + Y_grid**2)

ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Оригінальна функція")
plt.show()

for model_type, losses in loss_histories.items():
    plt.figure(figsize=(8, 5))
    for name, loss in losses.items():
        plt.plot(loss, label=name)
    plt.xlabel("Епохи")
    plt.ylabel("Втрати (Loss)")
    plt.title(f"Збіжність навчання ({model_type})")
    plt.legend()
    plt.show()

for model_type, models in architectures.items():
    for name, (layers, model) in models.items():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        with torch.no_grad():
            X_pred = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
            X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32)
            Z_pred = model(X_pred_tensor).numpy().reshape(X_grid.shape)

        ax.plot_surface(X_grid, Y_grid, Z_pred, cmap='viridis')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Апроксимація ({model_type} - {name})")
        plt.show()

plt.figure(figsize=(12, 6))
plt.bar(df_results["Модель"], df_results["Середня відносна помилка"])
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.ylabel("Середня відносна помилка")
plt.title("Помилка різних типів нейронних мереж")
plt.tight_layout()
plt.show()
