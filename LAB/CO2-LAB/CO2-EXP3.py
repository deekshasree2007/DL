import numpy as np
import matplotlib.pyplot as plt
 
np.random.seed(1)
X = np.array([[18, 1], [22, 2], [25, 3], [28, 4], [30, 5], [35, 6]], float)
y = (2.0 * X[:, 0] + 4.0 * X[:, 1] + 10).reshape(-1, 1)  # illustrative consumption
mu, sd = X.mean(axis=0), X.std(axis=0)
X = (X - mu) / sd
yn = (y - y.mean()) / y.std()
 
# One hidden layer, linear output
W1 = np.random.randn(2, 4) * 0.2
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.2
b2 = np.zeros((1, 1))
lr = 0.03
losses = []
 
for epoch in range(1500):
    h = np.tanh(X @ W1 + b1)
    out = h @ W2 + b2
    loss = np.mean((out - yn) ** 2)
    losses.append(loss)
    # Backpropagation for MSE
    d_out = 2 * (out - yn) / len(X)
    dW2 = h.T @ d_out
    db2 = d_out.sum(axis=0, keepdims=True)
    d_h = (d_out @ W2.T) * (1 - h ** 2)
    dW1 = X.T @ d_h
    db1 = d_h.sum(axis=0, keepdims=True)
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1
 
pred_n = np.tanh(X @ W1 + b1) @ W2 + b2
pred = pred_n * y.std() + y.mean()
print("Final normalized MSE:", losses[-1])
print("Actual consumption:", y.ravel().round(2))
print("Predicted consumption:", pred.ravel().round(2))
 
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.title("Electricity Consumption: Training Loss")
plt.grid(True)
plt.show()
