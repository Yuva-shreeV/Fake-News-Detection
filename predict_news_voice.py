import pickle, re, nltk, pyttsx3
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and TF-IDF
try:
    model = pickle.load(open("model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
except Exception as e:
    print("❌ Error loading model or vectorizer:", e)
    exit()

# Initialize NLTK
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 🔹 Clean & preprocess text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    return ' '.join(ps.stem(w) for w in text.split() if w not in stop_words)

# 🔹 Predict function
def predict_news(text):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "⚠️ Please enter meaningful text."
    vec = tfidf.transform([cleaned])
    return "Real News" if model.predict(vec)[0] == 1 else " Fake News"

# 🔹 Text-to-Speech
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.say(text)
    engine.runAndWait()

# 🔹 Main
if __name__ == "__main__":
    news = input("📝 Enter news text:\n")
    result = predict_news(news)
    print("🔍 Prediction:", result)
    speak(result)