import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Settings
# -----------------------------
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_PATH = "movie_embeddings.npy"
TITLES_PATH = "movie_titles.npy"
DATA_PATH = "imdb_top_1000.csv"

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

titles = df['Series_Title'].values
overviews = df['Overview'].fillna("").values

# Save movie titles
np.savetxt("movie_titles.txt", titles, fmt="%s")


# -----------------------------
# Tokenization
# -----------------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(overviews)
sequences = tokenizer.texts_to_sequences(overviews)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

print("Tokenization complete.")

# -----------------------------
# Build Simple RNN Encoder Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    tf.keras.layers.SimpleRNN(64),
])

print("Generating embeddings...")
embeddings = model.predict(padded_sequences, verbose=1)

np.save(EMBEDDING_PATH, embeddings)

print("Embeddings saved successfully!")
