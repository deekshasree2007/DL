#A medical decision-support system predicts whether a patient has a disease or not, with probabilities P = [0.9, 0.1]. Another patient's predicted probabilities are Q = [0.5, 0.5].
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Predicted probabilities [disease, no disease]
P = np.array([0.9, 0.1])     # Patient 1
Q = np.array([0.5, 0.5])     # Patient 2
 
# Step 2: Entropy function (bits)
def entropy(dist):
    return -np.sum(dist * np.log2(dist))
 
H_P = entropy(P)
H_Q = entropy(Q)
 
print("Entropy of Patient 1 prediction P =", round(H_P, 4), "bits")
print("Entropy of Patient 2 prediction Q =", round(H_Q, 4), "bits")
 
# Step 3: Identify the more uncertain prediction
if H_Q > H_P:
    print("Patient 2 (Q) has greater uncertainty (higher entropy).")
elif H_P > H_Q:
    print("Patient 1 (P) has greater uncertainty (higher entropy).")
else:
    print("Both predictions have equal uncertainty.")
 
# Step 4: Bar chart
labels = ["Patient 1 (P)", "Patient 2 (Q)"]
values = [H_P, H_Q]
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, values, color=["steelblue", "tomato"], edgecolor="black")
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
plt.ylabel("Entropy (bits)")
plt.title("Entropy of the Two Predictions")
plt.ylim(0, 1.2)
plt.show()
