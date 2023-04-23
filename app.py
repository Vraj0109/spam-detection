from flask import Flask, render_template, request
import pickle
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)
# Load the model


# Load the vectorizer
# import nltk
# nltk.download('stopwords')
# stop_words = list(stopwords.words('english'))
# vectorizer  = CountVectorizer(stop_words=stop_words)
model = pickle.load(open('D:\Study\sem 6\ML\lab_7\model.pkl', 'rb'))
vectorizer = pickle.load(open('D:\Study\sem 6\ML\lab_7\ectoriser.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the form
    text = request.form['text']

    # Preprocess the input
    input_data = vectorizer.transform([text])
    print(input_data.shape)

    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Render the result template with the prediction
    return render_template('index.html', prediction=prediction[0])
    # return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
