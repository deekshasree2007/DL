#A weather forecasting team develops prediction models with different polynomial degrees to estimate temperature based on time-related features. The team wants to identify a suitable model that balances simplicity and predictive accuracy.
import warnings
import numpy as np
import matplotlib.pyplot as plt
 
warnings.simplefilter("ignore")
np.random.seed(5)
 
# Step 1: Generate sample temperature data (hour of day 6 to 18, temperature in deg C)
n = 40
t = np.random.uniform(6, 18, n)
temp = 22 + 10 * np.sin(np.pi * (t - 6) / 12) + np.random.normal(0, 1, n)
x = (t - 12) / 6                                  # scale to [-1, 1]
 
# Step 2: Train / test split (70% / 30%)
idx = np.random.permutation(n)
train_idx, test_idx = idx[:28], idx[28:]
x_train, y_train = x[train_idx], temp[train_idx]
x_test, y_test = x[test_idx], temp[test_idx]
 
# Step 3: Train models and calculate errors
degrees = [1, 3, 12]
colors = ["red", "green", "purple"]
train_errors, test_errors = [], []
x_line = np.linspace(-1, 1, 400)
 
plt.figure(figsize=(9, 6))
plt.scatter(t[train_idx], y_train, color="blue", label="Training data")
plt.scatter(t[test_idx], y_test, color="orange", marker="s", label="Testing data")
 
print("Degree   Training MSE   Testing MSE")
for d, c in zip(degrees, colors):
    coef = np.polyfit(x_train, y_train, d)
    tr = np.mean((y_train - np.polyval(coef, x_train)) ** 2)
    te = np.mean((y_test - np.polyval(coef, x_test)) ** 2)
    train_errors.append(tr)
    test_errors.append(te)
    print(f"{d:6d}   {tr:12.4f}   {te:11.4f}")
    plt.plot(x_line * 6 + 12, np.polyval(coef, x_line), color=c, linewidth=2, label=f"Degree {d}")
 
best = degrees[int(np.argmin(test_errors))]
print("\nSuitable model (lowest testing MSE): Degree", best)
 
plt.ylim(10, 40)
plt.xlabel("Hour of the day")
plt.ylabel("Temperature (deg C)")
plt.title("Temperature Prediction: Degrees 1, 3 and 12")
plt.legend()
plt.grid(True)
plt.show()
 
# Step 4: Compare training and testing MSE
pos = np.arange(len(degrees))
width = 0.35
plt.figure(figsize=(7, 5))
plt.bar(pos - width / 2, train_errors, width, label="Training MSE", color="steelblue", edgecolor="black")
plt.bar(pos + width / 2, test_errors, width, label="Testing MSE", color="tomato", edgecolor="black")
plt.xticks(pos, [f"Degree {d}" for d in degrees])
plt.ylabel("Mean squared error")
plt.title("Training vs Testing Error")
plt.legend()
plt.show()
