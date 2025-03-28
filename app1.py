# -*- coding: utf-8 -*-
"""
Resume Screening - Machine Learning Project with Flask Deployment
"""

import numpy as np
import pandas as pd
import re
import warnings
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
file_path = r"C:\Users\WELCOME\Downloads\ks\resume_dataset.csv"
resumeDataSet = pd.read_csv(file_path, encoding='utf-8')

# Function to clean resume text
# Function to clean resume text
def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+', ' ', resume_text)  # Remove URLs
    resume_text = re.sub(r'RT|cc', ' ', resume_text)  # Remove RT and cc
    resume_text = re.sub(r'#\S+', '', resume_text)  # Remove hashtags
    resume_text = re.sub(r'@\S+', ' ', resume_text)  # Remove mentions
    resume_text = re.sub(r'[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', resume_text)  # Remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)  # Remove non-ASCII characters
    resume_text = re.sub(r'\s+', ' ', resume_text).strip()  # Remove extra whitespaces
    return resume_text

# Apply cleaning
if 'Resume' in resumeDataSet.columns:
    resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(clean_resume)
else:
    raise KeyError("Dataset does not contain 'Resume' column. Check your CSV file.")

# Encoding target variable
le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

# Feature extraction
word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=2000)
word_features = word_vectorizer.fit_transform(resumeDataSet['cleaned_resume'])

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(word_features, resumeDataSet['Category'], test_size=0.2, random_state=0)

# Model training using KNeighborsClassifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Save the model and vectorizer
pickle.dump(clf, open("model.pkl", "wb"))
pickle.dump(word_vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

# Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['resume']
    cleaned_data = clean_resume(data)
    vectorized_data = word_vectorizer.transform([cleaned_data])
    prediction = clf.predict(vectorized_data)
    category = le.inverse_transform(prediction)[0]
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)
