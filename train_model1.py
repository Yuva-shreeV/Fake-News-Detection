# train_model.py

import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize NLTK
nltk.download("stopwords", quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ---------- Text Cleaning ----------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower()
    return " ".join(ps.stem(w) for w in text.split() if w not in stop_words)

# ---------- Load Dataset ----------
df = pd.read_csv("dataset/combined_output.csv")
print("✅ Columns in dataset:", df.columns.tolist())

# ✅ Manually map your actual column names
# Replace 'content' and 'verdict' with your dataset's actual column names
df = df[["content", "verdict"]]  # ← Adjust these if needed
df.columns = ["text", "label"]   # Rename for consistency
df.dropna(subset=["text", "label"], inplace=True)

# ---------- Map Labels ----------
label_map = {
    "FAKE": 0, "fake": 0, "False": 0, "FALSE": 0, "F": 0, "0": 0,
    "REAL": 1, "real": 1, "True": 1, "TRUE": 1, "R": 1, "1": 1
}
df["label"] = df["label"].map(label_map).fillna(-1).astype(int)
df = df[df["label"].isin([0, 1])]

# ---------- Clean Text ----------
df["text"] = df["text"].apply(clean_text)

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ---------- TF-IDF Vectorization ----------
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ---------- Train Model ----------
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train_vec, y_train)

# ---------- Evaluate ----------
pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
print(f"\n✅ Accuracy: {acc:.3f}")
print(classification_report(y_test, pred, digits=3))

# ---------- Save Model & Vectorizer ----------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
print("✅ Saved model.pkl and tfidf.pkl")





