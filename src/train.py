import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
import os
from joblib import dump

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV and ensures it has the required columns.
    """
    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")
    return df

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing sets.
    """
    try:
        # Stratified split is preferred
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )
    except ValueError:
        # Fallback if stratification fails (e.g., on very small datasets)
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42
        )
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """
    Builds and trains a classification pipeline.
    """
    clf_pipeline = make_pipeline(
        TfidfVectorizer(min_df=1, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    clf_pipeline.fit(X_train, y_train)
    return clf_pipeline

# New function
def save_model(model: Pipeline, model_path: str) -> None:
    """
    Saves the trained model to a file.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)
    print(f"Saved model to {model_path}")

def main(data_path: str, model_path: str) -> None:
    """
    Main workflow to load, train, evaluate, and save the model.
    """
    df = load_and_validate_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    clf = train_model(X_train, y_train)

    # Evaluate and print accuracy
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")

    save_model(clf, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sentiments.csv")
    parser.add_argument("--out", default="models/sentiment.joblib")

    args: argparse.Namespace = parser.parse_args()
    main(data_path=args.data, model_path=args.out)