import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
 
np.random.seed(8)
X = np.random.rand(700, 3)  # lead time, previous attendance, reminder status
y = ((0.55*X[:,0] - 0.30*X[:,1] - 0.15*X[:,2] + 0.2) > 0.38).astype(int)
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
 
def build(dropout=False):
    seq = [layers.Input(shape=(3,)), layers.Dense(32, activation="relu")]
    if dropout:
        seq.append(layers.Dropout(0.4))
    seq += [layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")]
    m = keras.Sequential(seq)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m
 
results = {}
for name, drop, early in [
    ("Baseline", False, False), ("Dropout", True, False), ("EarlyStopping", False, True)]:
    model = build(drop)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                restore_best_weights=True)] if early else []
    h = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=60,
                  batch_size=32, callbacks=callbacks, verbose=0)
    results[name] = h.history
    print(name, "epochs run:", len(h.history["loss"]),
          "validation metrics:", model.evaluate(Xv, yv, verbose=0))
    np.savez(f"noshow_{name.lower()}.npz", **h.history)
 
for name, h in results.items():
    plt.plot(h["val_loss"], label=name)
plt.xlabel("Epoch"); plt.ylabel("Validation binary cross-entropy")
plt.title("Appointment No-Show: Validation Loss")
plt.legend(); plt.grid(True); plt.show()
