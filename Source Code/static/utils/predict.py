import tensorflow as tf
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import joblib


# Paths

model_path = 'C:/Users/pamid/ner_app/models/models/bodo_train_cnn_model.h5'
word_tokenizer_path = 'C:/Users/pamid/ner_app/models/word_tokenizer.pkl'
tag_tokenizer_path = 'C:/Users/pamid/ner_app/models/tag_tokenizer.pkl'

# Load model
model = tf.keras.models.load_model(model_path)

# Load tokenizers
with open(word_tokenizer_path, 'rb') as f:
    word_tokenizer = pickle.load(f)

with open(tag_tokenizer_path, 'rb') as f:
    tag_tokenizer = pickle.load(f)
max_length = 100
# Predict entities
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_entities(model, word_tokenizer, tag_tokenizer, sentence, max_length):
    # Tokenize and pad the input sentence
    
    sequence = word_tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Make predictions
    predictions = model.predict(padded_sequence)
    predicted_classes = np.argmax(predictions, axis=-1)

    # Map predicted classes to tags
    predicted_tags = [tag_tokenizer.index_word.get(tag, "O") for tag in predicted_classes[0]]

    return list(zip(sentence.split(), predicted_tags))  # Return word-tag pairs
