#A machine learning model uses eigenvectors to identify important directions in a two-dimensional dataset. The covariance matrix is C = [[3, 1], [1, 3]].
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Covariance matrix
C = np.array([[3, 1],
              [1, 3]])
 
# Step 2: Eigenvalues and eigenvectors (columns of 'eigenvectors')
eigenvalues, eigenvectors = np.linalg.eig(C)
 
print("Covariance matrix C:")
print(C)
print("\nEigenvalues :", eigenvalues)
print("\nEigenvectors (each column is an eigenvector):")
print(eigenvectors)
 
# Step 3: Plot eigenvectors as arrows from the origin
colors = ["red", "blue"]
plt.figure(figsize=(6, 6))
for i in range(2):
    v = eigenvectors[:, i]
    plt.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
               color=colors[i],
               label=f"v{i+1} = {np.round(v, 3)}, eigenvalue = {eigenvalues[i]:.2f}")
 
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.gca().set_aspect("equal")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Eigenvectors of Covariance Matrix C")
plt.legend(loc="lower right", fontsize=8)
plt.show()
