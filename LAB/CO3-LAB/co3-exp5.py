import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
 
tf.random.set_seed(5)
np.random.seed(5)
 
sales = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210], dtype="float32")
scale = 250.0
data = sales / scale
 
window = 3
X, y = [], []
for i in range(len(data) - window):
    X.append(data[i:i + window])
    y.append(data[i + window])
X = np.array(X).reshape(-1, window, 1)
y = np.array(y)
 
model = Sequential([
    Input(shape=(window, 1)),
    SimpleRNN(32, activation="linear"),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")
model.fit(X, y, epochs=400, verbose=0)
 
last = (sales[-3:] / scale).reshape(1, 3, 1)
pred = model.predict(last, verbose=0) * scale
print("Last 3 months : 190, 200, 210")
print("Forecast for next month (January):", round(float(pred[0][0]), 2))
