#A neural network classifies an image into four categories. The actual class distribution is Y = [0, 1, 0, 0], and the predicted probability distribution is P = [0.1, 0.6, 0.2, 0.1]. The developer wants to evaluate the prediction error and compare the predicted distribution with the actual distribution.
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Actual and predicted distributions
classes = ["Class 1", "Class 2", "Class 3", "Class 4"]
Y = np.array([0, 1, 0, 0])
P = np.array([0.1, 0.6, 0.2, 0.1])
 
# Step 2: Use only the terms where Y > 0 (because 0 * log(0) is taken as 0)
mask = Y > 0
 
# Cross-entropy  H(Y, P) = - sum( Y * ln(P) )
cross_entropy = -np.sum(Y[mask] * np.log(P[mask]))
 
# KL divergence  D_KL(Y || P) = sum( Y * ln(Y / P) )
kl_div = np.sum(Y[mask] * np.log(Y[mask] / P[mask]))
 
print("Actual distribution Y    :", Y)
print("Predicted distribution P :", P)
print("Cross-entropy loss       =", round(cross_entropy, 4))
print("KL divergence D_KL(Y||P) =", round(kl_div, 4))
print("(For a one-hot Y, the entropy of Y is 0, so cross-entropy = KL divergence.)")
 
# Step 3: Visualize the two distributions
x = np.arange(len(classes))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, Y, width, label="Actual (Y)", color="seagreen", edgecolor="black")
plt.bar(x + width / 2, P, width, label="Predicted (P)", color="orange", edgecolor="black")
plt.xticks(x, classes)
plt.ylabel("Probability")
plt.title(f"Actual vs Predicted (Cross-Entropy = {cross_entropy:.4f}, KL = {kl_div:.4f})")
plt.legend()
plt.show()
