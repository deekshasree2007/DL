#An online learning platform predicts student assessment scores using a machine learning model. The development team wants to understand whether the model suffers from high bias or high variance by examining its performance as the amount of training data increases
import warnings
import numpy as np
import matplotlib.pyplot as plt
 
warnings.simplefilter("ignore")
np.random.seed(7)
 
# Step 1: Generate the regression dataset (study hours -> assessment score)
def make_data(n):
    x = np.random.uniform(0, 10, n)
    y = 40 + 3 * x + 10 * np.sin(x) + np.random.normal(0, 3, n)
    return (x - 5) / 5, y                       # x scaled to [-1, 1]
 
x_train, y_train = make_data(100)               # training pool
x_val, y_val = make_data(100)                   # fixed validation set
 
# Step 2: Learning-curve function
def learning_curve(degree, sizes):
    train_err, val_err = [], []
    for m in sizes:
        coef = np.polyfit(x_train[:m], y_train[:m], degree)
        train_err.append(np.mean((y_train[:m] - np.polyval(coef, x_train[:m])) ** 2))
        val_err.append(np.mean((y_val - np.polyval(coef, x_val)) ** 2))
    return train_err, val_err
 
sizes = list(range(12, 101, 4))
 
# Step 3: Learning curves for a linear model and a high-degree polynomial model
lin_train, lin_val = learning_curve(1, sizes)
poly_train, poly_val = learning_curve(10, sizes)
 
print("Linear model (degree 1) with all 100 samples:")
print("   Training MSE   =", round(lin_train[-1], 2))
print("   Validation MSE =", round(lin_val[-1], 2))
print("Polynomial model (degree 10) with only 12 samples:")
print("   Training MSE   =", round(poly_train[0], 2))
print("   Validation MSE =", round(poly_val[0], 2))
print("Polynomial model (degree 10) with all 100 samples:")
print("   Training MSE   =", round(poly_train[-1], 2))
print("   Validation MSE =", round(poly_val[-1], 2))
 
# Step 4: Plot the learning curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
axes[0].plot(sizes, lin_train, "b-o", markersize=3, label="Training error")
axes[0].plot(sizes, lin_val, "r-o", markersize=3, label="Validation error")
axes[0].set_title("Linear Model (Degree 1): High Bias")
 
axes[1].plot(sizes, poly_train, "b-o", markersize=3, label="Training error")
axes[1].plot(sizes, poly_val, "r-o", markersize=3, label="Validation error")
axes[1].set_title("Polynomial Model (Degree 10): High Variance")
 
for ax in axes:
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Mean squared error")
    ax.legend()
    ax.grid(True)
 
axes[1].set_yscale("log")        # errors differ by orders of magnitude -> log scale
axes[1].set_ylabel("Mean squared error (log scale)")
 
plt.tight_layout()
plt.show()
