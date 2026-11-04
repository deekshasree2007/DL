import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
 
np.random.seed(4)
X = np.random.rand(300, 3) * 100  # aptitude, programming, communication
score = 0.35*X[:, 0] + 0.45*X[:, 1] + 0.20*X[:, 2]
y = (score >= 55).astype(int)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
 
def build_mlp(neurons):
    model = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(neurons, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
 
histories = {}
for neurons in [4, 16, 32]:
    model = build_mlp(neurons)
    h = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=40, batch_size=16, verbose=0)
    histories[neurons] = h.history
    print(neurons, "neurons - test metrics:", model.evaluate(X_val, y_val, verbose=0))
 
for neurons, h in histories.items():
    plt.plot(h["val_accuracy"], label=f"{neurons} neurons")
plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
plt.title("Placement MLP: Hidden Neuron Comparison")
plt.legend(); plt.grid(True); plt.show()
 
for neurons, h in histories.items():
    np.savez(f"placement_{neurons}_neurons.npz",
             accuracy=h["accuracy"], val_accuracy=h["val_accuracy"],
             loss=h["loss"], val_loss=h["val_loss"])
