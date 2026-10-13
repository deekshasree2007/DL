#A deep learning research team uses a machine learning model to classify images based on extracted numerical features. A highly complex polynomial model performs well on training data but poorly on unseen testing data.
import warnings
import numpy as np
import matplotlib.pyplot as plt
 
warnings.simplefilter("ignore")
np.random.seed(5)
 
# Step 1: Generate a non-linear dataset  y = sin(3x) + noise
n = 30
x = np.random.uniform(-1, 1, n)
y = np.sin(3 * x) + np.random.normal(0, 0.3, n)
 
# Step 2: Train / test split (60% / 40%)
idx = np.random.permutation(n)
train_idx, test_idx = idx[:18], idx[18:]
x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]
 
# Step 3: Train models of degree 2 and 15 and calculate errors
degrees = [2, 15]
colors = ["green", "red"]
x_line = np.linspace(-1, 1, 400)
 
plt.figure(figsize=(9, 6))
plt.scatter(x_train, y_train, color="blue", label="Training data")
plt.scatter(x_test, y_test, color="orange", marker="s", label="Testing data")
 
print("Degree   Training MSE   Testing MSE")
for d, c in zip(degrees, colors):
    coef = np.polyfit(x_train, y_train, d)
    train_mse = np.mean((y_train - np.polyval(coef, x_train)) ** 2)
    test_mse = np.mean((y_test - np.polyval(coef, x_test)) ** 2)
    print(f"{d:6d}   {train_mse:12.4f}   {test_mse:11.4f}")
    plt.plot(x_line, np.polyval(coef, x_line), color=c, linewidth=2, label=f"Degree {d}")
 
if test_mse > 2 * train_mse:
    print("\nThe degree-15 model has a very low training error but a much higher testing error -> overfitting.")
 
# Step 4: Compare the fitted curves
plt.ylim(-2, 2)
plt.xlabel("Feature value")
plt.ylabel("Output")
plt.title("Fitted Curves: Degree 2 vs Degree 15")
plt.legend()
plt.grid(True)
plt.show()
