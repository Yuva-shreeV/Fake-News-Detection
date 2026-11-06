import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
true_news = pd.read_csv("dataset/True.csv")
fake_news = pd.read_csv("dataset/Fake.csv")

# Add labels
true_news["label"] = 1   # Real = 1
fake_news["label"] = 0   # Fake = 0

# Combine datasets
data = pd.concat([true_news, fake_news], axis=0).reset_index(drop=True)

# 🔹 Use both title + text (better for short inputs)
data["content"] = data["title"].astype(str) + " " + data["text"].astype(str)

# Features and Labels
X = data["content"]
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 🔹 Model (LinearSVC works better for short texts)
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Training complete! Model saved as 'model.pkl' and 'tfidf.pkl'")






