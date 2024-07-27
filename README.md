# NextGenInterns_ML_03
Sentiment Analysis using Logistic regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
# Download necessary NLTK data
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('/content/IMDB Dataset.csv')

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove special characters
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    # Remove stopwords and perform stemming
    text = ' '.join(ps.stem(word) for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing to the dataset
df['review'] = df['review'].apply(preprocess_text)

# Split the dataset into training and testing sets
X = df['review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Build and train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)


