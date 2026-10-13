##A weather monitoring system records temperature and humidity over five days. The temperature values are [22, 25, 28, 30, 32], and the humidity values are [80, 75, 68, 60, 55].
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Data
temperature = np.array([22, 25, 28, 30, 32])
humidity = np.array([80, 75, 68, 60, 55])
 
# Step 2: Covariance matrix
cov_matrix = np.cov(temperature, humidity)
 
print("Covariance matrix:")
print(cov_matrix)
print("\nVariance of temperature :", cov_matrix[0, 0])
print("Variance of humidity    :", cov_matrix[1, 1])
print("Covariance (T, H)       :", cov_matrix[0, 1])
 
if cov_matrix[0, 1] < 0:
    print("Covariance is negative: humidity decreases as temperature increases.")
 
# Step 3: Scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(temperature, humidity, color="red", s=70)
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("Temperature vs Humidity")
plt.grid(True)
plt.show()
