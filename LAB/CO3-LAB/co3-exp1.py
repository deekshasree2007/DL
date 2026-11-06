import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
 
tf.random.set_seed(42)
np.random.seed(42)
 
# Study hours over four consecutive days: shape (samples, timesteps, features)
X = np.array([2, 3, 4, 5], dtype="float32").reshape(1, 4, 1)
 
model = Sequential([
    Input(shape=(4, 1)),
    SimpleRNN(5, activation="tanh"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.summary()
 
pred = model.predict(X, verbose=0)
print("Input sequence  :", X.flatten())
print("Predicted output:", pred[0][0])
