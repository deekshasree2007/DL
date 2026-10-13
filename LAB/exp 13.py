#A service centre receives an average of 4 customers every 10 minutes. The number of customer arrivals follows a Poisson distribution.
import math
import matplotlib.pyplot as plt
 
# Step 1: Parameters
lam = 4     # average number of arrivals in 10 minutes
 
# Step 2: Poisson PMF  P(X = k) = e^(-lambda) * lambda^k / k!
def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)
 
# Step 3: Probability of exactly 6 arrivals
prob_6 = poisson_pmf(6, lam)
print("P(exactly 6 customers) =", round(prob_6, 4))
 
# Step 4: PMF for 0 to 12 arrivals
k_values = list(range(0, 13))
pmf_values = [poisson_pmf(k, lam) for k in k_values]
 
# Step 5: Plot the PMF
colors = ["red" if k == 6 else "seagreen" for k in k_values]
plt.figure(figsize=(9, 5))
plt.bar(k_values, pmf_values, color=colors, edgecolor="black")
plt.xlabel("Number of customer arrivals (k)")
plt.ylabel("Probability P(X = k)")
plt.title("Poisson Distribution (average = 4 customers per 10 minutes)")
plt.xticks(k_values)
plt.show()
