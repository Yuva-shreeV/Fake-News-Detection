# predict_news.py

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# User Input
print("\n📰 Welcome to the Fake News Detector!")
news = input("👉 Paste your news article here:\n\n")

# Prediction
cleaned = clean_text(news)
vector = tfidf.transform([cleaned]).toarray()
prediction = model.predict(vector)

# Output
if prediction[0] == 1:
    print("\n🔴 This news is likely FAKE.")
else:
    print("\n🟢 This news is likely REAL.")
