#A weather prediction system estimates the probabilities of three possible conditions as sunny, rainy, and cloudy: P = [0.6, 0.3, 0.1]. The system needs to measure the uncertainty in its predictions.
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Probability distribution
conditions = ["Sunny", "Rainy", "Cloudy"]
P = np.array([0.6, 0.3, 0.1])
 
# Step 2: Entropy  H(P) = - sum( p * log2(p) )   (unit: bits)
entropy = -np.sum(P * np.log2(P))
 
print("Probabilities :", P)
print("Entropy H(P)  =", round(entropy, 4), "bits")
 
# Step 3: Bar chart of the probabilities
plt.figure(figsize=(7, 5))
bars = plt.bar(conditions, P, color=["gold", "royalblue", "gray"], edgecolor="black")
for bar, val in zip(bars, P):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, str(val), ha="center")
plt.ylabel("Probability")
plt.title(f"Weather Prediction Probabilities (Entropy = {entropy:.3f} bits)")
plt.ylim(0, 0.8)
plt.show()
