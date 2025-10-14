import argparse
import os
from typing import Tuple

import pandas as pd
from joblib import dump
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline


def load_and_validate_data(data_path: str) -> DataFrame:
    """
    Lädt Daten aus einer CSV, stellt sicher, dass die erforderlichen Spalten
    vorhanden sind, und konvertiert die Labels explizit in numerische Werte
    (0 und 1).
    """
    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV muss die Spalten 'text' und 'label' enthalten.")

    # Explizite Label-Konvertierung: 'positive' -> 1, 'negative' -> 0
    # (Wichtig für Stabilität)
    label_map = {"positive": 1, "negative": 0}
    df["label"] = df["label"].replace(label_map)

    return df


def train_model(X: Series, y: Series) -> Pipeline:
    """
    Baut und trainiert eine Klassifizierungspipeline mit extremen Parametern
    (C=100.0, max_iter=5000), um den winzigen Datensatz maximal zu überanpassen
    (Overfit).
    """
    clf_pipeline = make_pipeline(
        TfidfVectorizer(min_df=1, ngram_range=(1, 3)),
        # C=100.0 und hohe Iterationen erzwingen Overfitting
        LogisticRegression(max_iter=5000, C=100.0, class_weight="balanced"),
    )
    clf_pipeline.fit(X, y)
    return clf_pipeline


def evaluate_model(clf: Pipeline, X: Series, y: Series) -> None:
    """
    Bewertet das Modell per einfachem Score.
    """
    acc = clf.score(X, y)
    # Da wir absichtlich überanpassen, erwarten wir hier 1.0.
    print(f"Train/Evaluation Accuracy (Overfitting Check): {acc:.3f}")


def save_model(model: Pipeline, model_path: str) -> None:
    """
    Speichert das trainierte Modell in einer Datei.
    """
    # os.path.dirname wird von Python korrekt aufgelöst und ist Mypy-sauber
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)
    print(f"Saved model to {model_path}")


def main(data_path: str, model_path: str) -> None:
    """
    Haupt-Workflow: Lädt, trainiert auf allen Daten, bewertet (intern) und
    speichert das Modell.
    """
    df = load_and_validate_data(data_path)

    # Wir trainieren immer auf dem gesamten Datensatz, um die Konfidenz
    # für den Sanity Check zu maximieren.
    X: Series = df["text"]
    y: Series = df["label"]

    print(f"Dataset size ({len(df)}). Training on all data for high confidence test.")

    # Trainiere das Modell auf dem gesamten Datensatz (maximales Overfitting)
    clf = train_model(X, y)

    # Bewerte das Modell (wir erwarten hier 1.0, da wir überanpassen)
    evaluate_model(clf, X, y)

    save_model(clf, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sentiments.csv")
    parser.add_argument("--out", default="models/sentiment.joblib")

    args: argparse.Namespace = parser.parse_args()
    main(data_path=args.data, model_path=args.out)
