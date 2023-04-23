import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from autocorrect import Speller
import nltk
from flask import Flask, request, jsonify, render_template
import joblib

# Load the model and vectorizer from files
model = joblib.load("spam_classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Create a Flask web application
app = Flask(__name__)

# Define a route to serve the web page


@app.route("/")
def index():
    # Return the HTML from the file "index.html"
    return render_template("index.html")


# Define preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
spell = Speller(lang='en')


def preprocess(text):
    # Convert to lower case
    text = text.lower()
    # Remove stop words
    words = word_tokenize(text)
    words = [word for word in words if not word in stop_words]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    # Stem words
    # words = [stemmer.stem(word) for word in words]
    # Correct spelling mistakes
    words = [spell(word) for word in words]
    return ' '.join(words)

# Define a route to accept user input and return a prediction


@app.route("/predict", methods=["POST"])
def predict():
    # Get the user input
    message = request.form["message"]
    message = preprocess(message)
    print(message)
    # Preprocess the input
    message_vector = vectorizer.transform([message])

    # Make a prediction
    prediction = model.predict(message_vector)[0]

    # Return the prediction as JSON
    # return jsonify({"prediction": prediction})
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run()
