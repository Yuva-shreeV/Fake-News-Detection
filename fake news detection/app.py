import streamlit as st
import pickle, re, nltk, pyttsx3
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model & TF-IDF
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Initialize NLTK
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    return ' '.join(ps.stem(w) for w in text.split() if w not in stop_words)

# Predict
def predict_news(text):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "⚠️ Please enter meaningful text."
    vec = tfidf.transform([cleaned])
    return "Real News" if model.predict(vec)[0] == 1 else " Fake News"

# Text-to-speech
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news headline or article text to check if it's real or fake.")

news_input = st.text_area("✍️ Enter News Text Here")

if st.button("🔍 Predict"):
    result = predict_news(news_input)
    st.subheader("Result:")
    st.success(result if "🟢" in result else result)
    speak(result)  # optional
