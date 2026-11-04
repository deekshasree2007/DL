import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
 
# Folder structure: data/train/{healthy,diseased}, data/validation/{healthy,diseased}
train_ds = keras.utils.image_dataset_from_directory(
    "data/train", image_size=(64, 64), color_mode="grayscale",
    batch_size=16, label_mode="binary")
val_ds = keras.utils.image_dataset_from_directory(
    "data/validation", image_size=(64, 64), color_mode="grayscale",
    batch_size=16, label_mode="binary")
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))
 
model = keras.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
h = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=1)
print("Validation loss/accuracy:", model.evaluate(val_ds, verbose=0))
np.savez("leaf_cnn_metrics.npz", accuracy=h.history["accuracy"],
         val_accuracy=h.history["val_accuracy"], loss=h.history["loss"],
         val_loss=h.history["val_loss"])
 
plt.plot(h.history["accuracy"], label="Training")
plt.plot(h.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Leaf CNN Accuracy")
plt.legend(); plt.grid(True); plt.show()
 
# Illustration of a 4x4 grayscale matrix (not a substitute for the image dataset)
pixel_matrix = np.array([[10, 20, 20, 10], [20, 80, 90, 20],
                         [15, 70, 85, 15], [10, 20, 20, 10]], dtype=np.uint8)
plt.imshow(pixel_matrix, cmap="gray"); plt.title("Example 4x4 Grayscale Matrix")
plt.colorbar(); plt.show()
