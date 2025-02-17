import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
], dtype=torch.float32)

y = torch.tensor([
    [0], [1], [1], [0],
    [1], [0], [0], [1],
    [1], [0], [0], [1],
    [0], [1], [1], [0]
], dtype=torch.float32)

class XORPerceptron(nn.Module):
    def __init__(self):
        super(XORPerceptron, self).__init__()
        self.hidden = nn.Linear(4, 8)  
        self.output = nn.Linear(8, 1)  
        self.activation = nn.Sigmoid()  

    def forward(self, x):
        x = self.activation(self.hidden(x))  
        x = self.activation(self.output(x))  
        return x

model = XORPerceptron()
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.1)

epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

with torch.no_grad():
    predictions = model(X).round()  
    print("\nПеревірка результатів:")
    for i in range(len(X)):
        print(f"Вхід: {X[i].tolist()} -> Передбачення: {int(predictions[i].item())} | Очікуване: {int(y[i].item())}")
