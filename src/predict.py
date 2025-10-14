import argparse
import os
import sys

import joblib
import pandas as pd

# Pfad zum Modell relativ zum Container-Root (Wird nun durch Argumente überschrieben,
# aber als Fallback beibehalten)
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


def main(model_path: str, input_file: str, output_file: str):
    """Hauptfunktion zur Verarbeitung von Eingaben und zur Ausgabe von Vorhersagen."""

    model = load_model(model_path)

    # --- Neue Logik zur Unterscheidung von Text-Eingabe (sys.argv)
    # und Datei-Eingabe (Args) ---

    # Fall 1: Benutzer hat Text über die Kommandozeile übergeben
    # Wir prüfen sys.argv[1], da argparse die eigentlichen Argumente danach setzt.
    if len(sys.argv) > 1 and sys.argv[1] not in ["--help", "-h"]:
        # Wenn der erste Parameter KEIN Argument-Flag ist, behandeln wir es
        # als direkten Text
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
        # Fall 2: Dateimodus (Args wurden übergeben, z.B. --input ...)
        if not os.path.exists(input_file):
            print(f"[ERROR] Dateimodus ohne Eingabedatei: {input_file} nicht gefunden.")
            sys.exit(1)

        print(f"[INFO] Starte Vorhersage für {input_file}...")
        input_data = pd.read_csv(input_file)

        results_df = run_prediction(input_data, model)

        results_df.to_csv(output_file, index=False)
        print(f"[INFO] Vorhersagen erfolgreich gespeichert unter: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Wenn der ENTRYPOINT nur den Text übergibt (z.B. docker run app "Text"),
    # wird dieser in main() durch sys.argv[1] abgefangen. Die folgenden
    # Argumente sind für den Batch-Datei-Modus.
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--input", default="data/sentiments.csv")
    parser.add_argument("--output", default="data/sentiments_out.csv")

    args: argparse.Namespace = parser.parse_args()
    main(
        model_path=args.model,
        input_file=args.input,
        output_file=args.output,
    )
