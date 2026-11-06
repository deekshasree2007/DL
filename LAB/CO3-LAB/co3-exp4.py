import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
 
tf.random.set_seed(3)
np.random.seed(3)
 
text = "machine learning is interesting"
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(chars)
seq_len = 4
 
X, y = [], []
for i in range(len(text) - seq_len):
    X.append([char_to_idx[c] for c in text[i:i + seq_len]])
    y.append(char_to_idx[text[i + seq_len]])
 
X = to_categorical(X, num_classes=vocab_size)   # (samples, 4, vocab)
y = to_categorical(y, num_classes=vocab_size)
 
model = Sequential([
    Input(shape=(seq_len, vocab_size)),
    SimpleRNN(64),
    Dense(vocab_size, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=300, verbose=0)
 
def generate(seed, n_chars=10):
    result = seed
    for _ in range(n_chars):
        x = np.array([[char_to_idx[c] for c in result[-seq_len:]]])
        x = to_categorical(x, num_classes=vocab_size)
        nxt = np.argmax(model.predict(x, verbose=0))
        result += idx_to_char[nxt]
    return result
 
print("Generated text:", generate("mach", 12))
