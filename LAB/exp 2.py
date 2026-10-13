#A deep learning model predicts student performance based on study hours, practice-test scores, and assignment scores. The input features are X = [4, 6, 8], the corresponding weights are W = [0.5, 0.3, 0.2], and the bias is b = 2.import numpy as np
 
import numpy as np
 
# Step 1: Inputs, weights and bias
X = np.array([4, 6, 8])        # study hours, practice-test score, assignment score
W = np.array([0.5, 0.3, 0.2])  # weights
b = 2                          # bias
 
# Step 2: Dot product of X and W
dot_product = np.dot(X, W)
 
# Step 3: Add the bias
y_pred = dot_product + b
 
# Step 4: Display the results
print("Input vector X   :", X)
print("Weight vector W  :", W)
print("Bias b           :", b)
print("Dot product X.W  :", dot_product)
print("Predicted output :", y_pred)
