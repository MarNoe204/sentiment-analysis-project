import os
from unittest.mock import patch # <- Wichtig: Jetzt wieder benötigt, um F401 zu vermeiden und CLI zu mocken

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
    # Wir müssen sys.argv mocken, damit predict_main die --input und --output Argumente
    # korrekt liest und nicht die Pytest-Argumente verwendet, die zum FileNotFoundError führen.
    temp_file_name = "temp_out.csv"
    with patch(
        "sys.argv",
        [
            "src/predict.py",
            "--input",
            TEST_DATA_PATH,
            "--output",
            temp_file_name,
        ],
    ):
        # Wichtig: Wir übergeben die Argumente auch direkt, aber das Mocking ist
        # für das CLI-Parsing notwendig.
        predict_main(
            model_path=MODEL_PATH, input_file=TEST_DATA_PATH, output_file=temp_file_name
        )

    # 3. Vorhersageergebnisse laden
    results_df = pd.read_csv(temp_file_name)
    predictions_int = results_df["label"].tolist()

    # 4. Die numerischen Vorhersagen in Text-Labels umwandeln
    predictions_str = [
        "positive" if pred == 1 else "negative" for pred in predictions_int
    ]

    # 5. Die erwarteten Labels ebenfalls in Strings umwandeln
    expected_labels_str = [
        "positive" if label == 1 else "negative" for label in expected_labels_int
    ]

    # 6. Überprüfung
    assert len(predictions_str) == len(expected_labels_str)
    assert predictions_str == expected_labels_str

    # 7. Cleanup der temporären Ausgabedatei
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)
