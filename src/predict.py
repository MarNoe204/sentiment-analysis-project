import joblib
import os
import sys

import pandas as pd

# Pfad zum Modell relativ zum Container-Root
MODEL_PATH = "models/sentiment.joblib"


def load_model(path: str):
    """Lädt das vorab trainierte Modell aus der Joblib-Datei."""
    try:
        model = joblib.load(path)
        print(f"[INFO] Modell erfolgreich geladen von: {path}")
        return model
    except FileNotFoundError:
        print(
            f"[ERROR] Modelldatei nicht gefunden unter: {path}. "
            "Bitte trainieren Sie das Modell zuerst."
        )
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden des Modells: {e}")
        sys.exit(1)


def run_prediction(input_data: pd.DataFrame, model) -> pd.DataFrame:
    """Führt die Sentiment-Analyse auf den Eingabedaten durch."""
    # Extrahieren Sie die Textspalte, auf der das Modell trainiert wurde
    # (Wir gehen davon aus, dass die erste Spalte (Index 0) der Text ist)
    X = input_data.iloc[:, 0]

    # Vorhersage des Sentiments (Label und Konfidenz)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Bestimme die maximale Konfidenz (Wahrscheinlichkeit)
    confidence = [max(p) for p in probabilities]

    # Füge die Ergebnisse zum DataFrame hinzu
    input_data["label"] = predictions
    input_data["confidence"] = confidence

    return input_data


def main():
    """Hauptfunktion zur Verarbeitung von Eingaben und zur Ausgabe von Vorhersagen."""

    model = load_model(MODEL_PATH)

    # --- Neue Logik zur Unterscheidung von Datei- und Text-Eingabe ---
    if len(sys.argv) > 1:
        # Fall 1: Benutzer hat Text über die Kommandozeile übergeben (ENTRYPOINT-Nutzung)
        input_text = sys.argv[1]

        # Erstelle einen DataFrame aus dem einzelnen String
        input_data = pd.DataFrame([input_text], columns=["text"])

        print(f"[INFO] Analysiere Text: '{input_text[:50]}...'")

        # Führe die Vorhersage aus
        results_df = run_prediction(input_data, model)

        # Gebe das Ergebnis auf der Konsole aus
        print("\n--- Analyse Ergebnis ---")
        print(results_df[["text", "label", "confidence"]].iloc[0].to_markdown())
        print("-----------------------\n")

    else:
        # Fall 2: Keine Argumente übergeben, führe Dateiverarbeitung aus
        # (Standardverhalten)
        input_file = "data/sentiments.csv"
        output_file = "data/sentiments_out.csv"

        if not os.path.exists(input_file):
            print(f"[ERROR] Dateimodus ohne Eingabedatei: {input_file} nicht gefunden.")
            sys.exit(1)

        print(f"[INFO] Starte Vorhersage für {input_file}...")
        input_data = pd.read_csv(input_file)

        results_df = run_prediction(input_data, model)

        results_df.to_csv(output_file, index=False)
        print(f"[INFO] Vorhersagen erfolgreich gespeichert unter: {output_file}")


if __name__ == "__main__":
    main()
