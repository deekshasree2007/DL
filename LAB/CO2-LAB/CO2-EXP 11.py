import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from tensorflow.keras import layers
 
# Folder structure: data/train/{deer,elephant,tiger}, data/validation/{deer,elephant,tiger}
train_ds = keras.utils.image_dataset_from_directory(
    "data/train", image_size=(64, 64), batch_size=32, label_mode="int")
val_ds = keras.utils.image_dataset_from_directory(
    "data/validation", image_size=(64, 64), batch_size=32, label_mode="int")
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))
 
def build(deep=False):
    seq = [layers.Input(shape=(64, 64, 3)),
           layers.Conv2D(16, 3, activation="relu", padding="same"),
           layers.MaxPooling2D()]
    if deep:
        seq += [layers.Conv2D(32, 3, activation="relu", padding="same"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, activation="relu", padding="same"),
                layers.MaxPooling2D()]
    else:
        seq += [layers.Conv2D(32, 3, activation="relu", padding="same"),
                layers.MaxPooling2D()]
    seq += [layers.Flatten(), layers.Dense(64, activation="relu"),
            layers.Dense(3, activation="softmax")]
    m = keras.Sequential(seq)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
 
histories = {}
for name, deep in [("Shallow CNN", False), ("Deeper CNN", True)]:
    model = build(deep)
    start = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=8, verbose=1)
    elapsed = time.time() - start
    histories[name] = h.history
    print(name, "training seconds:", round(elapsed, 2),
          "validation loss/accuracy:", model.evaluate(val_ds, verbose=0))
    np.savez(name.replace(" ", "_") + ".npz", training_seconds=elapsed, **h.history)
 
for name, h in histories.items():
    plt.plot(h["val_accuracy"], marker="o", label=name)
plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
plt.title("Wildlife CNN Architecture Comparison")
plt.legend(); plt.grid(True); plt.show()
