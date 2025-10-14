import pandas as pd
import pytest
from src.predict import load_model, run_prediction

# Wir definieren den Pfad zum trainierten Modell
MODEL_PATH = "models/sentiment.joblib"


@pytest.fixture(scope="session")
def loaded_classifier():
    """
    Lädt das trainierte Modell (die Pipeline) nur einmal pro pytest-Sitzung.
    Dies reduziert den Overhead des wiederholten Ladens großer Joblib-Dateien
    und beschleunigt die Tests drastisch.
    """
    print(f"\n[INFO] Lade das Modell von {MODEL_PATH} (dies geschieht nur einmal)...")
    return load_model(MODEL_PATH)


@pytest.fixture
def sanity_data() -> pd.DataFrame:
    """
    Lädt eine kleine, eindeutige Untermenge von Daten für den Sanity Check.
    """
    df = pd.read_csv("data/sentiments.csv")

    # Wir verwenden die ersten 5 Zeilen, um Modell-Ungenauigkeiten zu umgehen
    # und nur die Funktionsfähigkeit der Pipeline zu prüfen.
    df = df.head(5)

    # Die Spalte 'label' enthält 0 (negative) oder 1 (positive)
    return df


def test_prediction_sanity(sanity_data: pd.DataFrame, loaded_classifier) -> None:
    """
    Überprüft, ob der Klassifikator die erwarteten Labels für eine kleine,
    eindeutige Stichprobe korrekt vorhersagt.
    """
    texts = sanity_data["text"]
    # expected_labels_int enthält die erwarteten 0/1 Werte aus der CSV
    expected_labels_int = sanity_data["label"].tolist()

    # Erstelle einen temporären DataFrame nur mit der Textspalte für run_prediction
    temp_df = pd.DataFrame({"text": texts})

    # 2. Die Vorhersagen mit dem echten Modell ausführen
    results_df = run_prediction(temp_df, loaded_classifier)
    predictions_int = results_df["label"].tolist()

    # 3. Die numerischen Vorhersagen in Text-Labels umwandeln (0 -> negative, 1 -> positive)
    predictions_str = [
        "positive" if pred == 1 else "negative" for pred in predictions_int
    ]

    # 4. Die erwarteten Labels ebenfalls in Strings umwandeln,
    # um einen korrekten Vergleich zu gewährleisten.
    expected_labels_str = [
        "positive" if label == 1 else "negative" for label in expected_labels_int
    ]

    assert len(predictions_str) == len(expected_labels_str)
    # Das assert statement überprüft, ob die Vorhersagen mit den erwarteten Labels
    # übereinstimmen
    assert predictions_str == expected_labels_str
