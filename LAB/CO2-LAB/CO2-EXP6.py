import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
 
np.random.seed(5)
X = np.random.rand(500, 3)  # pressure, temperature, production speed
y = ((0.4*X[:,0] + 0.35*X[:,1] + 0.25*X[:,2]) > 0.52).astype(int)
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
 
def build_model(deep=False):
    model = keras.Sequential([layers.Input(shape=(3,)), layers.Dense(16, activation="relu")])
    if deep:
        model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
 
results = {}
for name, deep in [("Single hidden layer", False), ("Multiple hidden layers", True)]:
    model = build_model(deep)
    h = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=35, batch_size=32, verbose=0)
    results[name] = h.history
    print(name, "final validation loss/accuracy:",
          model.evaluate(Xv, yv, verbose=0))
    np.savez(name.replace(" ", "_") + ".npz", **h.history)
 
for name, h in results.items():
    plt.plot(h["val_accuracy"], label=name)
plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
plt.title("Product Quality: Architecture Comparison")
plt.legend(); plt.grid(True); plt.show()
