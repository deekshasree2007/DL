#A recommendation system represents two movies using numerical feature vectors. The first movie is represented by A = [2, 4, 6], and the second movie is represented by B = [1, 3, 5]. The system combines the feature vectors to create a combined representation and calculates the similarity score using their dot product.
import numpy as np
 
# Step 1: Movie feature vectors
A = np.array([2, 4, 6])
B = np.array([1, 3, 5])
 
# Step 2: Vector addition (combined representation)
combined = A + B
 
# Step 3: Dot product (similarity score)
similarity = np.dot(A, B)
 
# Step 4: Display the results
print("Movie A features :", A)
print("Movie B features :", B)
print("Vector addition (A + B) :", combined)
print("Dot product (similarity score) :", similarity)
