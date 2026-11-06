import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
 
tf.random.set_seed(1)
np.random.seed(1)
 
X = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50],
              [40, 50, 60], [50, 60, 70]], dtype="float32")
y = np.array([40, 50, 60, 70, 80], dtype="float32")
 
# Scale values for stable training
X_s, y_s = X / 100.0, y / 100.0
X_s = X_s.reshape(5, 3, 1)
 
model = Sequential([
    Input(shape=(3, 1)),
    SimpleRNN(32, activation="linear"),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")
model.fit(X_s, y_s, epochs=400, verbose=0)
 
test = np.array([60, 70, 80], dtype="float32").reshape(1, 3, 1) / 100.0
pred = model.predict(test, verbose=0) * 100.0
print("Input     : 60, 70, 80")
print("Predicted :", round(float(pred[0][0]), 2))
