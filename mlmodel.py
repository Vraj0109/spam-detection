import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the dataset
emails = pd.read_csv("D:\Study\sem 6\ML\lab_7\spam.csv", encoding='latin-1')
emails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
emails.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)


# Preprocessing
import nltk
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
count_vector = CountVectorizer(stop_words=stop_words)
email_counts = count_vector.fit_transform(emails['message'].values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(email_counts, emails['label'].values, test_size=0.3, random_state=42)

# Train the model
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Confusion Matrix: ", confusion)
with open('D:\Study\sem 6\ML\lab_7\model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

with open('D:\Study\sem 6\ML\lab_7\ectoriser.pkl', 'wb') as file:
    pickle.dump(count_vector, file)

str = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
str = count_vector.transform([str])

print(classifier.predict(str))
