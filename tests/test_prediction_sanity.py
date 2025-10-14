import os
from unittest.mock import patch

import pandas as pd
import pytest

from src import train
from src.predict import main as predict_main  # predict_main um Konflikte zu vermeiden

# Feste Konstanten
TEST_DATA_PATH = "data/test_sentiments.csv"
MODEL_PATH = "models/test_sentiment.joblib"


# --- Fixtures ---
@pytest.fixture(scope="session")
def setup_test_data():
    """Erstellt temporäre Testdaten für Training und Sanity Check."""
    test_data = pd.DataFrame(
        {
            "text": [
                "I absolutely love this product and would buy it again.",
                "This is the worst experience, I feel totally scammed.",
                "It's okay, nothing special but it works.",
                "Simply fantastic, five stars all the way!",
                "Utterly disappointed with the quality and service.",
            ],
            # 1=positive, 0=negative (wie vom train.py erwartet)
            "label": [1, 0, 0, 1, 0],
        }
    )
    test_data.to_csv(TEST_DATA_PATH, index=False)
    # Liefere die numerischen Labels zur Überprüfung
    expected_labels = test_data["label"].tolist()
    yield expected_labels
    # Cleanup
    os.remove(TEST_DATA_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)


# --- Sanity Test ---
def test_prediction_sanity(setup_test_data):
    """
    Testet, ob das trainierte Modell auf dem winzigen Trainingsdatensatz (Overfit)
    eine perfekte Vorhersage (Sanity Check) liefert.
    """
    expected_labels_int = setup_test_data

    # 1. Modell auf den Testdaten trainieren (erzeugt Overfitting)
    train.main(data_path=TEST_DATA_PATH, model_path=MODEL_PATH)

    # 2. Vorhersage mit der predict_main-Funktion ausführen
    with patch(
        "sys.argv",
        ["src/predict.py", "--input", TEST_DATA_PATH, "--output", "temp_out.csv"],
    ):
        predict_main(
            model_path=MODEL_PATH, input_file=TEST_DATA_PATH, output_file="temp_out.csv"
        )

    results_df = pd.read_csv("temp_out.csv")
    predictions_int = results_df["label"].tolist()

    # 3. Die numerischen Vorhersagen in Text-Labels umwandeln (0 -> negative, 
    # 1 -> positive)
    predictions_str = [
        "positive" if pred == 1 else "negative" for pred in predictions_int
    ]

    # 4. Die erwarteten Labels ebenfalls in Strings umwandeln, um einen korrekten
    # Vergleich zu gewährleisten.
    expected_labels_str = [
        "positive" if label == 1 else "negative" for label in expected_labels_int
    ]

    # 5. Überprüfung
    assert len(predictions_str) == len(expected_labels_str)
    # Das assert statement überprüft, ob die Vorhersagen mit den erwarteten
    # Labels übereinstimmen
    assert predictions_str == expected_labels_str

    # Cleanup der temporären Ausgabedatei
    if os.path.exists("temp_out.csv"):
        os.remove("temp_out.csv")
