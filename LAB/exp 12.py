#A lecturer observes that students' examination marks approximately follow a normal distribution with a mean of 70 and a standard deviation of 10.
import math
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Parameters
mu = 70
sigma = 10
a, b = 60, 80
 
# Step 2: Normal CDF using the error function
def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
 
# Step 3: P(60 < X < 80) = F(80) - F(60)
prob = normal_cdf(b, mu, sigma) - normal_cdf(a, mu, sigma)
print("P(60 < X < 80) =", round(prob, 4))
print("Percentage     =", round(prob * 100, 2), "%")
 
# Step 4: Normal probability density curve
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
 
# Step 5: Plot with the interval highlighted
plt.figure(figsize=(9, 5))
plt.plot(x, pdf, color="blue", label="Normal distribution")
plt.fill_between(x, pdf, where=(x >= a) & (x <= b), color="orange", alpha=0.6,
                 label=f"P({a} < X < {b}) = {prob:.4f}")
plt.xlabel("Marks")
plt.ylabel("Probability density")
plt.title("Normal Distribution (mean = 70, standard deviation = 10)")
plt.legend()
plt.grid(True)
plt.show()
