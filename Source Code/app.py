
from flask import Flask, render_template, request, jsonify
from utils.tokenize import tokenize_text
from utils.predict import predict_entities
import os

import tensorflow as tf
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# Paths to model and tokenizers
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

max_length = 100  # Define maximum length for padding/truncation

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Predictions route
@app.route('/predictions', methods=['POST'])
def predictions():
    if request.method == 'POST':
        # Extract text from form
        text = request.form['text']
        
        # Predict entities
        entities = predict_entities(model, word_tokenizer, tag_tokenizer, text, max_length)

        # Render predictions in the template
        return render_template('predictions.html', tokens=text.split(), entities=entities)

# Metrics route
@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

# Flowchart route
@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

if __name__ == '__main__':
    app.run(debug=True)
