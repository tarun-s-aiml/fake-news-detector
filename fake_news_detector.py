import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "news.csv"


def load_dataset():

    try:
        data = pd.read_csv(DATASET_PATH)

    except FileNotFoundError:
        print(f"Dataset not found: {DATASET_PATH}")
        return None

    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        print("Dataset must contain 'text' and 'label' columns.")
        return None

    data = data.dropna(subset=["text", "label"])

    return data


def train_model(data):

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)

    print("\n===== MODEL PERFORMANCE =====")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, predictions))

    return model, vectorizer


def predict_news(model, vectorizer):

    print("\n===== FAKE NEWS DETECTOR =====")

    news = input("Enter news text: ").strip()

    if not news:
        print("Please enter some news text.")
        return

    news_vectorized = vectorizer.transform([news])

    prediction = model.predict(news_vectorized)[0]

    print("\n===== PREDICTION =====")
    print("Result:", prediction)


def main():

    print("===== FAKE NEWS DETECTION SYSTEM =====")

    data = load_dataset()

    if data is None:
        return

    print(f"Dataset size: {len(data)} records")

    model, vectorizer = train_model(data)

    predict_news(model, vectorizer)


if __name__ == "__main__":
    main()
