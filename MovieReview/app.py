import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential

# Load the IMDB dataset word index

word_index=imdb.get_word_index()
reverse_word_index={value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation

model=load_model('simple_rnn_imdb.h5')

# Functions to decode reviews

def decode_review(encode_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in emcoded_review])

# Function to preprocess user input

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded-review],maxlen=500)
    return padded_review

## Prediction Function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(prepocessesed_input)

    sentiment='positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment, prediction[0][0]

## Streamlit app

import streamlit as st

st.title("IMDB Moview Sentiment Analysis")
st.write('Enter a moview review to classify it as positive or negative.')

# User Input

user_input=st.text_area("moview Review")

if st.button('classify'):
    preprocess_input=predict(preprocess_input)

    #Make prediction

    prediction=model.predict(prepocesses_input)
    sentiment='positive' if prediction[0][0]>0.5 else 'Negative'

    # Display the result
    
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('please enter a moview review')
    
