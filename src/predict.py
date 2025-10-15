import argparse
import os
import sys

import joblib
import pandas as pd

# Pfad zum Modell relativ zum Container-Root (Wird nun durch Argumente überschrieben, 
# aber als Fallback beibehalten)
MODEL_PATH = 'models/sentiment.joblib'


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
    input_data['label'] = predictions
    input_data['confidence'] = confidence

    return input_data


def run_cli_text_mode(input_text: str, model_path: str = MODEL_PATH):
    """
    Führt die Vorhersage für einen direkt über die Kommandozeile übergebenen Text aus.
    Dieser Modus umgeht argparse.
    """
    model = load_model(model_path)
    
    # Erstelle einen DataFrame aus dem einzelnen String
    input_data = pd.DataFrame([input_text], columns=['text'])

    print(f"[INFO] Analysiere Text: '{input_text[:50]}...'")

    # Führe die Vorhersage aus
    results_df = run_prediction(input_data, model)

    # Gebe das Ergebnis auf der Konsole aus (wichtig für den Docker Sanity Check!)
    print("\n--- Analyse Ergebnis ---")
    # Der Output für den Docker Check muss auf der Konsole sein, ohne to_csv
    # Wir geben die relevanten Daten direkt aus, um den grep-Befehl im CI zu vereinfachen.
    # Format: label, confidence
    result_series = results_df[['label', 'confidence']].iloc[0]
    label_text = 'positive' if result_series['label'] == 1 else 'negative'
    
    # Wichtig: Drucke die Daten, die der grep-Befehl in der CI erwartet (label und confidence)
    # Beispiel-Output: "Sentiment: positive | Confidence: 0.9876"
    print(f"Sentiment: {label_text} | Confidence: {result_series['confidence']:.4f}")
    
    # Gib den Markdown-Output für eine schöne Anzeige aus (für den Menschen)
    print("\n" + results_df[['text', 'label', 'confidence']].iloc[0].to_markdown())
    print("-----------------------\n")


def main(model_path: str, input_file: str, output_file: str):
    """Hauptfunktion für den Batch-Modus (Datei-zu-Datei-Verarbeitung)."""

    model = load_model(model_path)

    if not os.path.exists(input_file):
        print(f"[ERROR] Eingabedatei nicht gefunden: {input_file}")
        sys.exit(1)

    print(f"[INFO] Starte Vorhersage für Batch-Datei: {input_file}...")
    input_data = pd.read_csv(input_file)

    results_df = run_prediction(input_data, model)

    results_df.to_csv(output_file, index=False)
    print(f"[INFO] Vorhersagen erfolgreich gespeichert unter: {output_file}")


if __name__ == "__main__":
    # --- Modus-Erkennung (bevor argparse läuft!) ---
    
    # Prüft, ob ein Argument übergeben wurde, das NICHT mit einem Flag beginnt.
    # Das ist der Fall bei: docker run app "some text"
    if len(sys.argv) > 1 and not sys.argv[1].startswith(("-", "--")):
        # Ausführung im Text-Modus und sofortiger Exit
        run_cli_text_mode(sys.argv[1])
        sys.exit(0) # Erfolgreicher Exit nach Text-Modus-Ausführung
        
    # --- Batch-Modus (File-to-File) ---
    # Fällt zurück auf argparse, wenn keine Text-Eingabe gefunden wurde (oder --help übergeben wurde)
    
    parser = argparse.ArgumentParser(
        description="Führt eine Sentiment-Analyse in Batch- oder Text-Modus durch."
    )
    
    parser.add_argument("--model", default=MODEL_PATH, help="Pfad zur gespeicherten Modell-Datei.")
    parser.add_argument("--input", default="data/sentiments.csv", help="Pfad zur Eingabe-CSV-Datei.")
    parser.add_argument("--output", default="data/sentiments_out.csv", help="Pfad zur Ausgabe-CSV-Datei.")

    args: argparse.Namespace = parser.parse_args()
    main(
        model_path=args.model,
        input_file=args.input,
        output_file=args.output,
    )
