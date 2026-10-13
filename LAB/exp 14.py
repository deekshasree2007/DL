#A disease affects 2% of a population. A medical test correctly identifies a person with the disease 95% of the time. The test also produces a positive result for 5% of people who do not have the disease.
import matplotlib.pyplot as plt
 
# Step 1: Given probabilities
p_disease = 0.02              # P(D)   prior probability
p_pos_given_disease = 0.95    # P(+|D) sensitivity
p_pos_given_healthy = 0.05    # P(+|not D) false positive rate
 
# Step 2: Probability of no disease
p_healthy = 1 - p_disease
 
# Step 3: Total probability of a positive test
p_positive = p_pos_given_disease * p_disease + p_pos_given_healthy * p_healthy
 
# Step 4: Bayes' theorem  P(D|+) = P(+|D) * P(D) / P(+)
posterior = (p_pos_given_disease * p_disease) / p_positive
 
print("Prior probability P(D)            =", p_disease)
print("Total probability of positive P(+) =", round(p_positive, 4))
print("Posterior probability P(D | +)     =", round(posterior, 4))
print("Posterior in percentage            =", round(posterior * 100, 2), "%")
 
# Step 5: Compare prior and posterior
labels = ["Prior P(D)", "Posterior P(D | +)"]
values = [p_disease, posterior]
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, values, color=["steelblue", "tomato"], edgecolor="black")
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.4f}", ha="center")
plt.ylabel("Probability")
plt.title("Probability of Disease: Before and After a Positive Test")
plt.ylim(0, max(values) + 0.08)
plt.show()
