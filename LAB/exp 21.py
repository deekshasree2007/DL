#A real estate company develops machine learning models to predict house prices based on house area. A linear regression model and polynomial regression mod
#els of degrees 2 and 10 are trained using the same dataset. The company wants to compare how model complexity affects prediction performance.
import warnings
import numpy as np
import matplotlib.pyplot as plt
 
warnings.simplefilter("ignore")      # hide numerical warnings for high-degree fits
np.random.seed(1)
 
# Step 1: Generate sample house-price data (area in sq.ft, price in lakhs)
n = 30
area = np.sort(np.random.uniform(500, 2500, n))
price = 30 + 0.05 * area - 0.000012 * area ** 2 + np.random.normal(0, 6, n)
 
# Step 2: Scale the area to [-1, 1] (keeps high-degree fitting stable)
x = (area - 1500) / 1000
 
# Step 3: Train polynomial models of degrees 1, 2 and 10 and plot them
degrees = [1, 2, 10]
colors = ["red", "green", "purple"]
x_line = np.linspace(x.min(), x.max(), 300)
 
plt.figure(figsize=(9, 6))
plt.scatter(area, price, color="black", label="Actual data")
 
for d, c in zip(degrees, colors):
    coef = np.polyfit(x, price, d)                       # train the model
    train_mse = np.mean((price - np.polyval(coef, x)) ** 2)
    print(f"Degree {d:2d} -> Training MSE = {train_mse:.2f}")
    plt.plot(x_line * 1000 + 1500, np.polyval(coef, x_line), color=c, linewidth=2,
             label=f"Degree {d}")
 
plt.ylim(price.min() - 20, price.max() + 20)
plt.xlabel("House area (sq.ft)")
plt.ylabel("House price (lakhs)")
plt.title("Polynomial Regression: Degrees 1, 2 and 10")
plt.legend()
plt.grid(True)
plt.show()
