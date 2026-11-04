import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from tensorflow.keras import layers
 
# Folder structure: data/train/{plastic,paper,metal}, data/validation/{plastic,paper,metal}
train_ds = keras.utils.image_dataset_from_directory(
    "data/train", image_size=(64, 64), batch_size=32, label_mode="int")
val_ds = keras.utils.image_dataset_from_directory(
    "data/validation", image_size=(64, 64), batch_size=32, label_mode="int")
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))
 
def build(filters=16, lr=0.001):
    m = keras.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(filters, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(filters*2, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="softmax")
    ])
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
 
# Vary one setting at a time from the baseline
configs = [
    ("baseline", 16, 0.001, 32),
    ("low_lr", 16, 0.0003, 32),
    ("high_lr", 16, 0.003, 32),
    ("more_filters", 32, 0.001, 32),
    ("small_batch", 16, 0.001, 16)
]
rows = []
for name, filters, lr, batch in configs:
    # Rebuild datasets with matching batch size
    tr = keras.utils.image_dataset_from_directory(
        "data/train", image_size=(64,64), batch_size=batch, label_mode="int")
    va = keras.utils.image_dataset_from_directory(
        "data/validation", image_size=(64,64), batch_size=batch, label_mode="int")
    tr = tr.map(lambda x, y: (x/255.0, y))
    va = va.map(lambda x, y: (x/255.0, y))
    model = build(filters, lr)
    start = time.time()
    h = model.fit(tr, validation_data=va, epochs=6, verbose=0)
    elapsed = time.time() - start
    val_loss, val_acc = model.evaluate(va, verbose=0)
    rows.append([name, val_acc, val_loss, elapsed])
    print(name, "val_accuracy:", round(val_acc, 4),
          "val_loss:", round(val_loss, 4), "seconds:", round(elapsed, 2))
 
names = [r[0] for r in rows]
val_acc = np.array([r[1] for r in rows])
val_loss = np.array([r[2] for r in rows])
seconds = np.array([r[3] for r in rows])
np.savez("waste_hyperparameter_results.npz", names=names,
         val_accuracy=val_acc, val_loss=val_loss, training_seconds=seconds)
 
plt.bar(names, val_acc)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Validation accuracy"); plt.title("Waste CNN Hyperparameter Comparison")
plt.tight_layout(); plt.show()
