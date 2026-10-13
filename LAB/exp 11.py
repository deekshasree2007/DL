#A factory produces electronic components, and the probability that a component is defective is 0.1. A quality inspector randomly selects 20 components. The number of defective components follows a binomial distribution
import math
import matplotlib.pyplot as plt
 
# Step 1: Parameters
n = 20      # number of components inspected
p = 0.1     # probability that a component is defective
 
# Step 2: Binomial PMF  P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
def binomial_pmf(k, n, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
 
# Step 3: Probability of exactly 3 defective components
prob_3 = binomial_pmf(3, n, p)
print("P(exactly 3 defective) =", round(prob_3, 4))
 
# Step 4: PMF for 0 to 20 defective components
k_values = list(range(0, n + 1))
pmf_values = [binomial_pmf(k, n, p) for k in k_values]
print("Sum of all probabilities =", round(sum(pmf_values), 4))
 
# Step 5: Plot the PMF
colors = ["red" if k == 3 else "steelblue" for k in k_values]
plt.figure(figsize=(9, 5))
plt.bar(k_values, pmf_values, color=colors, edgecolor="black")
plt.xlabel("Number of defective components (k)")
plt.ylabel("Probability P(X = k)")
plt.title("Binomial Distribution (n = 20, p = 0.1)")
plt.xticks(k_values)
plt.show()
