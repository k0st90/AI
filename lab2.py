import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

# Генерація даних
def generate_data(n_samples=1500):
    x = np.random.uniform(0, 10, n_samples)
    y = np.random.uniform(0, 10, n_samples)
    z = np.exp(np.sin(x) + np.cos(y)) + np.log(1 + x**2 + y**2)  # Цільова функція
    return np.column_stack((x, y)), z

# Перетворення у тензори
def prepare_tensors(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X_tensor, y_tensor

# Моделі нейронних мереж
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
            self.hidden_states.append(torch.zeros(1, h))  # Початковий контекстний стан
            prev_size = h

        self.output_layer = nn.Linear(prev_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]  # Отримуємо batch size
        inputs = x
        new_states = []
        for i, layer in enumerate(self.hidden_layers):
            hidden_expanded = self.hidden_states[i].expand(batch_size, -1)
            inputs = torch.cat((inputs, hidden_expanded), dim=1)
            inputs = torch.relu(layer(inputs))

            # Оновлення стану без збереження градієнтів
            new_states.append(inputs.mean(dim=0, keepdim=True).detach())

        self.hidden_states = new_states  # Оновлення станів
        return self.output_layer(inputs)

# Функція для тренування
def train_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    y_pred_test = model(X_test).detach().numpy()
    error = mean_absolute_error(y_test, y_pred_test) / np.mean(y_test.numpy())
    return error

# Генерація даних
X, y = generate_data()
X_train, y_train = prepare_tensors(X[:1000], y[:1000])
X_test, y_test = prepare_tensors(X[1000:], y[1000:])

# Архітектури мереж
architectures = {
    "FeedForward (10 нейронів)": ([10], FeedForwardNN(2, [10])),
    "FeedForward (20 нейронів)": ([20], FeedForwardNN(2, [20])),
    "CascadeForward (1 шар, 20 нейронів)": ([20], CascadeForwardNN(2, [20])),
    "CascadeForward (2 шари, 10 нейронів)": ([10, 10], CascadeForwardNN(2, [10, 10])),
    "Elman (1 шар, 15 нейронів)": ([15], ElmanNN(2, [15])),
    "Elman (3 шари, 5 нейронів)": ([5, 5, 5], ElmanNN(2, [5, 5, 5]))
}

# Тренування моделей та оцінка помилки
results = []
for name, (layers, model) in architectures.items():
    error = train_model(model, X_train, y_train, X_test, y_test)
    results.append({"Модель": name, "Середня відносна помилка": error})

# Відображення результатів
df_results = pd.DataFrame(results)
print(df_results)

# Візуалізація результатів
plt.figure(figsize=(10, 5))
plt.bar(df_results["Модель"], df_results["Середня відносна помилка"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Середня відносна помилка")
plt.title("Помилка різних типів нейронних мереж")
plt.show()
