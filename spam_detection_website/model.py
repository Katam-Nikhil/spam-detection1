"""Train and save the SMS/email spam classification model."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "spam.csv"
MODEL_FILE = BASE_DIR / "spam_model.pkl"

def train_model():
    data = pd.read_csv(DATA_FILE)
    required = {"category", "message"}
    if not required.issubset(data.columns):
        raise ValueError("spam.csv must contain 'category' and 'message' columns.")

    data = data.dropna(subset=["category", "message"])
    data["category"] = data["category"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str)
    data = data[data["category"].isin(["spam", "ham"])]

    if data["category"].nunique() < 2:
        raise ValueError("The dataset must contain both spam and ham examples.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
        ("classifier", MultinomialNB())
    ])
    pipeline.fit(data["message"], data["category"])
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model trained with {len(data)} messages.")
    print(f"Saved model to: {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
