#A deep learning model classifies an image into one of three categories: cat, dog, or bird. The actual class is represented by Y = [1, 0, 0], while the model predicts probabilities P = [0.7, 0.2, 0.1].
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Actual and predicted distributions
classes = ["Cat", "Dog", "Bird"]
Y = np.array([1, 0, 0])            # actual class (one-hot)
P = np.array([0.7, 0.2, 0.1])      # predicted probabilities
 
# Step 2: Cross-entropy loss  H(Y, P) = - sum( Y * ln(P) )
cross_entropy = -np.sum(Y * np.log(P))
 
print("Actual distribution Y    :", Y)
print("Predicted distribution P :", P)
print("Cross-entropy loss       =", round(cross_entropy, 4))
 
# Step 3: Grouped bar chart comparing actual and predicted
x = np.arange(len(classes))
width = 0.35
plt.figure(figsize=(7, 5))
plt.bar(x - width / 2, Y, width, label="Actual (Y)", color="seagreen", edgecolor="black")
plt.bar(x + width / 2, P, width, label="Predicted (P)", color="orange", edgecolor="black")
plt.xticks(x, classes)
plt.ylabel("Probability")
plt.title(f"Actual vs Predicted Probabilities (Cross-Entropy = {cross_entropy:.4f})")
plt.legend()
plt.show()
