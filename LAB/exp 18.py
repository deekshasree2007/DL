#Two language models predict the next word using the probability distributions P = [0.5, 0.3, 0.2] and Q = [0.4, 0.4, 0.2]. A developer wants to measure how much the second model's distribution differs from the first.
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Probability distributions of the two models
words = ["Word 1", "Word 2", "Word 3"]
P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.4, 0.4, 0.2])
 
# Step 2: KL divergence  D_KL(P || Q) = sum( P * ln(P / Q) )
kl_pq = np.sum(P * np.log(P / Q))
kl_qp = np.sum(Q * np.log(Q / P))      # reverse direction (for comparison)
 
print("P =", P)
print("Q =", Q)
print("KL divergence D_KL(P || Q) =", round(kl_pq, 4))
print("KL divergence D_KL(Q || P) =", round(kl_qp, 4), "(KL divergence is not symmetric)")
 
# Step 3: Grouped bar chart
x = np.arange(len(words))
width = 0.35
plt.figure(figsize=(7, 5))
plt.bar(x - width / 2, P, width, label="Model 1 (P)", color="steelblue", edgecolor="black")
plt.bar(x + width / 2, Q, width, label="Model 2 (Q)", color="tomato", edgecolor="black")
plt.xticks(x, words)
plt.ylabel("Probability")
plt.title(f"Comparison of P and Q (KL Divergence = {kl_pq:.4f})")
plt.legend()
plt.show()
