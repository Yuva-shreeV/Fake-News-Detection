# predict_news.py
import pickle, re, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and TF-IDF
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Initialize NLTK
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Clean & preprocess text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    return ' '.join(ps.stem(w) for w in text.split() if w not in stop_words)

# Predict function
def predict_news(text):
    vec = tfidf.transform([clean_text(text)])
    return "🟢 Real News" if model.predict(vec)[0] == 1 else "🔴 Fake News"

# Run
if __name__ == "__main__":
    news = input("Enter news text:\n")
    print("Prediction:", predict_news(news))

