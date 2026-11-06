import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
 
tf.random.set_seed(7)
np.random.seed(7)
 
reviews = ["I love this product", "Excellent quality", "Very useful product",
           "I hate this product", "Very poor quality", "Product is terrible"]
labels = np.array([1, 1, 1, 0, 0, 0])      # 1 = Positive, 0 = Negative
 
tokenizer = Tokenizer(num_words=50, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
max_len = max(len(s) for s in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding="post")
vocab_size = len(tokenizer.word_index) + 1
 
model = Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 8),
    SimpleRNN(8),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, labels, epochs=200, verbose=0)
loss, acc = model.evaluate(X, labels, verbose=0)
print("Training accuracy:", round(acc, 2))
 
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post")
    p = model.predict(pad, verbose=0)[0][0]
    return ("Positive" if p > 0.5 else "Negative"), float(p)
 
for review in ["I love this quality", "Product is very poor"]:
    label, p = predict_sentiment(review)
    print(f"Review: '{review}' -> {label} (score = {p:.3f})")
