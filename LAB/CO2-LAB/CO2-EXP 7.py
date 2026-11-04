import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
 
np.random.seed(7)
X = np.random.rand(600, 4) * [2500, 5, 40, 30]  # area, rooms, age, distance
y = (100 + 0.08*X[:,0] + 12*X[:,1] - 0.7*X[:,2] - 2*X[:,3]
     + np.random.randn(600)*10).astype("float32")
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
scx = StandardScaler()
Xtr = scx.fit_transform(Xtr); Xv = scx.transform(Xv)
ymean, ystd = ytr.mean(), ytr.std()
ytr = (ytr-ymean)/ystd; yv = (yv-ymean)/ystd
 
def build(kind):
    reg = regularizers.l2(0.01) if kind == "L2" else None
    model = keras.Sequential([layers.Input(shape=(4,)),
        layers.Dense(64, activation="relu", kernel_regularizer=reg)])
    if kind == "Dropout":
        model.add(layers.Dropout(0.35))
    model.add(layers.Dense(32, activation="relu", kernel_regularizer=reg))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model
 
histories = {}
for kind in ["Baseline", "L2", "Dropout"]:
    model = build(kind)
    h = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=60,
                  batch_size=32, verbose=0)
    histories[kind] = h.history
    print(kind, "final validation MSE:", h.history["val_loss"][-1])
    np.savez(f"house_{kind.lower()}.npz", loss=h.history["loss"],
             val_loss=h.history["val_loss"])
 
for kind, h in histories.items():
    plt.plot(h["val_loss"], label=kind)
plt.xlabel("Epoch"); plt.ylabel("Validation MSE")
plt.title("House Price: Regularization Comparison")
plt.legend(); plt.grid(True); plt.show()
s
