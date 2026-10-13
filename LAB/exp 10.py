#An intelligent monitoring system collects measurements from two sensors. The covariance matrix of the measurements is C = [[5, 2], [2, 2]]. The system uses eigenvalue and eigenvector analysis to identify the principal direction of variation in the sensor data.
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Covariance matrix of the two sensors
C = np.array([[5, 2],
              [2, 2]])
 
# Step 2: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(C)
 
print("Covariance matrix C:")
print(C)
print("\nEigenvalues :", eigenvalues)
print("\nEigenvectors (columns):")
print(eigenvectors)
 
# Step 3: Principal direction = eigenvector of the largest eigenvalue
idx = np.argmax(eigenvalues)
principal = eigenvectors[:, idx]
print("\nLargest eigenvalue :", eigenvalues[idx])
print("Principal direction (eigenvector) :", np.round(principal, 4))
angle = np.degrees(np.arctan2(principal[1], principal[0]))
print("Angle of principal direction with x-axis :", round(angle, 2), "degrees")
 
# Step 4: Plot both eigenvectors as arrows
plt.figure(figsize=(6, 6))
for i in range(2):
    v = eigenvectors[:, i]
    is_principal = (i == idx)
    plt.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
               color="red" if is_principal else "blue",
               label=("Principal eigenvector" if is_principal else "Second eigenvector")
                     + f" (eigenvalue = {eigenvalues[i]:.2f})")
 
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.gca().set_aspect("equal")
plt.grid(True)
plt.xlabel("Sensor 1 direction")
plt.ylabel("Sensor 2 direction")
plt.title("Eigenvectors and Principal Direction of Variation")
plt.legend(loc="lower right", fontsize=8)
plt.show()
