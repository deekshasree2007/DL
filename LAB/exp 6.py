#A college analyzes the relationship between students' study hours and examination marks to understand how study time is associated with academic performance. The study hours are [2, 4, 6, 8, 10], and the corresponding marks are [35, 45, 60, 75, 85].
import numpy as np
import matplotlib.pyplot as plt
 
# Step 1: Data
hours = np.array([2, 4, 6, 8, 10])
marks = np.array([35, 45, 60, 75, 85])
n = len(hours)
 
# Step 2: Sample covariance (manual) = sum((x - mean_x)(y - mean_y)) / (n - 1)
cov_manual = np.sum((hours - hours.mean()) * (marks - marks.mean())) / (n - 1)
 
# Step 3: Sample covariance using NumPy (ddof=1 -> sample covariance)
cov_numpy = np.cov(hours, marks, ddof=1)[0, 1]
 
print("Mean of study hours       :", hours.mean())
print("Mean of marks             :", marks.mean())
print("Sample covariance (manual):", cov_manual)
print("Sample covariance (NumPy) :", cov_numpy)
 
if cov_numpy > 0:
    print("Covariance is positive: marks increase as study hours increase.")
 
# Step 4: Scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(hours, marks, color="blue", s=70)
plt.xlabel("Study Hours")
plt.ylabel("Examination Marks")
plt.title("Study Hours vs Examination Marks")
plt.grid(True)
