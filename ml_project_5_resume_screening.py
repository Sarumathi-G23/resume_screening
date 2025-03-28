# -*- coding: utf-8 -*-
"""
Resume Screening - Machine Learning Project
"""

import numpy as np
import pandas as pd
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
file_path = r"C:\Users\WELCOME\Downloads\ks\resume_dataset.csv"
resumeDataSet = pd.read_csv(file_path, encoding='utf-8')

# Display dataset info
print("Displaying the distinct categories of resume:")
print(resumeDataSet['Category'].unique())
print("\nCategory distribution:")
print(resumeDataSet['Category'].value_counts())

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
print("\nCategory Encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# Feature extraction
word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=2000)
word_features = word_vectorizer.fit_transform(resumeDataSet['cleaned_resume'])

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(word_features, resumeDataSet['Category'], test_size=0.2, random_state=0)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Model training using KNeighborsClassifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Model evaluation
prediction = clf.predict(X_test)
print(f'Accuracy on training set: {clf.score(X_train, y_train):.2f}')
print(f'Accuracy on test set: {clf.score(X_test, y_test):.2f}')
print("\nClassification Report:")
print(metrics.classification_report(y_test, prediction))
