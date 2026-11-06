import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
 
tf.random.set_seed(2)
np.random.seed(2)
 
sentences = ["I love machine learning", "I love deep learning",
             "I love artificial intelligence", "I like machine learning",
             "I like deep learning"]
 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
 
# Build (prefix -> next word) training pairs
X_seq, y = [], []
for s in sentences:
    tokens = tokenizer.texts_to_sequences([s])[0]
    for i in range(1, len(tokens)):
        X_seq.append(tokens[:i])
        y.append(tokens[i])
 
max_len = max(len(s) for s in X_seq)
X = pad_sequences(X_seq, maxlen=max_len, padding="pre")
y = to_categorical(y, num_classes=vocab_size)
 
model = Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 10),
    SimpleRNN(32),
    Dense(vocab_size, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=300, verbose=0)
 
def predict_next(phrase):
    seq = tokenizer.texts_to_sequences([phrase])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    probs = model.predict(seq, verbose=0)[0]
    return tokenizer.index_word[int(np.argmax(probs))], probs
 
word, probs = predict_next("I love")
print("Input: 'I love'  -> Predicted next word:", word)
for i in np.argsort(probs)[::-1][:4]:
    if i != 0:
        print(f"   {tokenizer.index_word[int(i)]:<12} {probs[i]:.3f}")
