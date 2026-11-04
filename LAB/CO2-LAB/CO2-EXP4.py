import numpy as np
import matplotlib.pyplot as plt
 
np.random.seed(2)
# [soil moisture, fertilizer]; high yield = 1
X = np.array([[10, 1], [15, 2], [20, 2], [35, 5], [40, 6], [45, 7]], float)
y = np.array([[0], [0], [0], [1], [1], [1]], float)
X = X / X.max(axis=0)
 
W1 = np.random.randn(2, 5) * 0.3
b1 = np.zeros((1, 5))
W2 = np.random.randn(5, 1) * 0.3
b2 = np.zeros((1, 1))
lr = 0.2
losses = []
 
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
 
def forward(X):
    h = np.tanh(X @ W1 + b1)
    p = sigmoid(h @ W2 + b2)
    return h, p
 
_, p0 = forward(X)
print("Before training:", (p0 >= 0.5).astype(int).ravel())
 
for epoch in range(1500):
    h, p = forward(X)
    loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
    losses.append(loss)
    dz2 = (p - y) / len(X)
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    dh = (dz2 @ W2.T) * (1 - h**2)
    dW1 = X.T @ dh
    db1 = dh.sum(axis=0, keepdims=True)
    W2 -= lr*dW2; b2 -= lr*db2
    W1 -= lr*dW1; b1 -= lr*db1
 
_, p1 = forward(X)
print("After training:", (p1 >= 0.5).astype(int).ravel())
print("Final binary cross-entropy:", losses[-1])
 
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.title("Crop Yield Classification Loss")
plt.grid(True)
plt.show()
