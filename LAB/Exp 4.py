#A handwriting recognition model adjusts its weights during training to reduce prediction errors. The current weight vector is W = [0.7, 0.5, 0.9], the gradient vector is G = [0.2, 0.1, 0.3], and the learning rate is 0.1.
import numpy as np
 
# Step 1: Current weights, gradient and learning rate
W = np.array([0.7, 0.5, 0.9])
G = np.array([0.2, 0.1, 0.3])
eta = 0.1
 
# Step 2: Gradient descent update rule  W_new = W - eta * G
W_new = W - eta * G
 
# Step 3: Display the results
print("Old weight vector     :", W)
print("Gradient vector       :", G)
print("Learning rate         :", eta)
print("Updated weight vector :", W_new)
