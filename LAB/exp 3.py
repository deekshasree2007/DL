#A weather prediction system receives temperature and humidity measurements from two locations. The input matrix is X = [[30, 60], [35, 70]]. The model uses the weight matrix W = [[0.2, 0.5], [0.4, 0.3]].

import numpy as np
 
# Step 1: Input matrix (rows = locations, columns = temperature, humidity)
X = np.array([[30, 60],
              [35, 70]])
 
# Step 2: Weight matrix
W = np.array([[0.2, 0.5],
              [0.4, 0.3]])
 
# Step 3: Matrix multiplication
output = np.matmul(X, W)       # same as X @ W
 
# Step 4: Display the results
print("Input matrix X:")
print(X)
print("\nWeight matrix W:")
print(W)
print("\nPredicted output matrix (X x W):")
print(output)
print("\nLocation 1 output :", output[0])
print("Location 2 output :", output[1])
