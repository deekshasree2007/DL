import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow import keras
from tensorflow.keras import layers
 
# Folder structure: data/train/{speed_limit,stop,no_entry}, data/validation/{speed_limit,stop,no_entry}
train_ds = keras.utils.image_dataset_from_directory(
    "data/train", image_size=(64, 64), batch_size=32, label_mode="int")
val_ds = keras.utils.image_dataset_from_directory(
    "data/validation", image_size=(64, 64), batch_size=32, label_mode="int")
class_names = train_ds.class_names
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))
 
model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h = model.fit(train_ds, validation_data=val_ds, epochs=10)
print("Validation loss/accuracy:", model.evaluate(val_ds, verbose=0))
 
y_true = np.concatenate([y.numpy() for _, y in val_ds])
prob = model.predict(val_ds, verbose=0)
y_pred = np.argmax(prob, axis=1)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("Class order:", class_names)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
plt.title("Road Sign Confusion Matrix"); plt.show()
 
np.savez("road_sign_metrics.npz", accuracy=h.history["accuracy"],
         val_accuracy=h.history["val_accuracy"], loss=h.history["loss"],
         val_loss=h.history["val_loss"], confusion_matrix=cm)
plt.plot(h.history["accuracy"], label="Train accuracy")
plt.plot(h.history["val_accuracy"], label="Validation accuracy")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Road Sign Accuracy"); plt.grid(True); plt.show()
