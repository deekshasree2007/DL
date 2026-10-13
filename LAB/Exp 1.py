#A hospital uses a deep learning model to classify medical images. A small grayscale image is represented by a 3  3 matrix of pixel intensities. Before classification, the image must be converted into a one-dimensional vector, and the total pixel intensity must be calculated. 
import numpy as np
 
# Step 1: Image matrix (3 x 3 grayscale image)
A = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])
 
# Step 2: Flatten the matrix into a 1-D vector
flat_vector = A.flatten()
 
# Step 3: Sum of all pixel intensities
total_intensity = np.sum(flat_vector)
 
# Step 4: Display the results
print("Image matrix A:")
print(A)
print("Flattened vector :", flat_vector)
print("Sum of all pixel intensities :", total_intensity)
