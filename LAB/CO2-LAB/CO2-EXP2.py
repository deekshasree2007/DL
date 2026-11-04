import numpy as np
import matplotlib.pyplot as plt
 
# [temperature, vibration]; 1 = inspection required
X = np.array([[30, 1], [35, 2], [40, 2], [65, 6], [70, 7], [80, 8]], float)
y = np.array([0, 0, 0, 1, 1, 1])
Xn = X / X.max(axis=0)
 
def forward(x, w, b):
    z = x @ w + b
    return z, (z >= 0).astype(int)
 
w = np.array([0.2, 0.2])
b = -0.7
z, pred = forward(Xn, w, b)
print("Weighted sums:", z)
print("Predicted labels:", pred)
print("Actual labels:   ", y)
print("Initial accuracy:", np.mean(pred == y))
 
# Perceptron update
lr = 0.1
for epoch in range(40):
    for xi, target in zip(Xn, y):
        _, out = forward(xi, w, b)
        error = target - int(out)
        w += lr * error * xi
        b += lr * error
 
z, pred = forward(Xn, w, b)
print("Updated weights:", w, "bias:", b)
print("Final predictions:", pred, "accuracy:", np.mean(pred == y))
 
plt.scatter(Xn[y == 0, 0], Xn[y == 0, 1], label="Continue (0)")
plt.scatter(Xn[y == 1, 0], Xn[y == 1, 1], label="Inspect (1)")
xx = np.linspace(0, 1, 100)
if abs(w[1]) > 1e-8:
    plt.plot(xx, -(w[0] * xx + b) / w[1], label="Decision boundary")
plt.xlabel("Normalized temperature")
plt.ylabel("Normalized vibration")
plt.title("Industrial Machine Status")
plt.legend()
plt.grid(True)
plt.show()
