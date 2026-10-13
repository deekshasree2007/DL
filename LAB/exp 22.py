#A college develops a machine learning model to predict students' examination marks based on study hours. The model uses a simple linear relationship, but the actual marks follow a nonlinear pattern.
import numpy as np
import matplotlib.pyplot as plt
 
np.random.seed(0)
 
# Step 1: Generate non-linear student-performance data (S-shaped relationship)
n = 60
hours = np.random.uniform(0, 10, n)
marks = 100 / (1 + np.exp(-(hours - 5))) + np.random.normal(0, 4, n)
 
# Step 2: Split into training (70%) and testing (30%) sets
idx = np.random.permutation(n)
train_idx, test_idx = idx[:42], idx[42:]
x_train, y_train = hours[train_idx], marks[train_idx]
x_test, y_test = hours[test_idx], marks[test_idx]
 
# Step 3: Train a linear regression model  (marks = m * hours + c)
coef = np.polyfit(x_train, y_train, 1)
print("Slope =", round(coef[0], 3), " Intercept =", round(coef[1], 3))
 
# Step 4: Predictions and mean squared errors
pred_train = np.polyval(coef, x_train)
pred_test = np.polyval(coef, x_test)
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print("Training MSE =", round(train_mse, 2))
print("Testing MSE  =", round(test_mse, 2))
print("Both errors are high -> the linear model underfits the non-linear data.")
 
# Step 5: Compare actual and predicted marks (first 5 test samples)
print("\nStudy hours   Actual marks   Predicted marks")
for h, a, p in list(zip(x_test, y_test, pred_test))[:5]:
    print(f"{h:11.2f}   {a:12.2f}   {p:15.2f}")
 
# Plot 1: data and fitted line
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color="blue", label="Training data")
plt.scatter(x_test, y_test, color="orange", label="Testing data")
x_line = np.linspace(0, 10, 100)
plt.plot(x_line, np.polyval(coef, x_line), color="red", linewidth=2, label="Linear model")
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.title("Linear Regression on Non-linear Data")
plt.legend()
plt.grid(True)
plt.show()
 
# Plot 2: actual vs predicted marks (test set)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_test, color="purple")
plt.plot([0, 100], [0, 100], "k--", label="Perfect prediction")
plt.xlabel("Actual marks")
plt.ylabel("Predicted marks")
plt.title("Actual vs Predicted Marks (Test Set)")
plt.legend()
plt.grid(True)
plt.show()
