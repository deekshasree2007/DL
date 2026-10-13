#An image-processing system uses the matrix A = [[4, 2], [2, 4]] to analyze patterns in image features.
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Matrix
A = np.array([[4, 2],
              [2, 4]])
 
# Step 2: Eigenvalues
eigenvalues = np.linalg.eigvals(A)
eigenvalues = np.sort(eigenvalues)[::-1]      # largest first
 
print("Matrix A:")
print(A)
print("\nEigenvalues:", eigenvalues)
 
# Step 3: Bar chart
labels = ["Eigenvalue 1", "Eigenvalue 2"]
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, eigenvalues, color=["steelblue", "orange"], edgecolor="black")
for bar, val in zip(bars, eigenvalues):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.2f}", ha="center")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalues of Matrix A")
plt.show()
