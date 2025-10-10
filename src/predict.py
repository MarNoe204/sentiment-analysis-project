import argparse
from typing import Any
import numpy as np
from numpy.typing import NDArray
from joblib import load
import pandas as pd # NEU: Import für Dateiverarbeitung
import os           # NEU: Import für Pfadoperationen

# Pfad zum Modell, der als Standardwert verwendet wird
MODEL_PATH = "models/sentiment.joblib"

def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    # HINWEIS: Es ist ratsam, hier einen try/except Block hinzuzufügen, 
    # um FileNotFoundError zu behandeln, falls die Datei fehlt.
    return load(model_path)


def predict_texts(
    classifier: Any, input_texts: list[str]
) -> tuple[list[int], list[float | None]]:
    """Return labels and probability-of-positive for each text."""
    preds: NDArray[Any] = classifier.predict(input_texts)
    if hasattr(classifier, "predict_proba"):
        # Stelle sicher, dass input_texts mindestens ein Element hat, um Indexfehler zu vermeiden
        if not input_texts:
            return [], []

        probs_arr: NDArray[np.float64] = classifier.predict_proba(input_texts)[:, 1]
        probs = [float(p) for p in probs_arr.tolist()]
    else:
        probs = [None] * len(input_texts)
    return preds.astype(int).tolist(), probs


def format_prediction_lines(
    texts: list[str], preds: list[int], probs: list[float | None]
) -> list[str]:
    """Return tab-separated CLI output lines for each input text."""
    lines: list[str] = []
    for text, pred, prob in zip(texts, preds, probs):
        # Angenommen, 1=positive, 0=negative für die Ausgabe
        label = "positive" if pred == 1 else "negative"
        if prob is None:
            lines.append(f"{label}\t{text}")
        else:
            lines.append(f"{label}\t{prob:.3f}\t{text}")
    return lines


# NEUE FUNKTION: Wird für den Docker CMD Befehl benötigt
def run_file_prediction(input_path: str, output_path: str, model_path: str = MODEL_PATH) -> None:
    """Führt die Vorhersage für eine CSV-Datei durch und speichert das Ergebnis."""
    print(f"[INFO] Starte Vorhersage für {input_path}...")

    # 1. Daten laden
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[FEHLER] Fehler beim Laden der Input-Datei: {e}")
        return

    if 'text' not in df.columns:
        print("[FEHLER] Die Input-CSV muss eine Spalte namens 'text' enthalten.")
        return

    texts = df['text'].astype(str).tolist()

    # 2. Modell laden und Vorhersage ausführen
    classifier = load_model(model_path)
    predictions_int, predictions_proba = predict_texts(classifier, texts)
    
    # 3. Ergebnisse hinzufügen
    df['label'] = predictions_int
    df['sentiment'] = ["positive" if pred == 1 else "negative" for pred in predictions_int]
    df['confidence'] = predictions_proba
    
    # 4. Speichern der Ergebnisse
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[INFO] Vorhersagen erfolgreich gespeichert unter: {output_path}")


def main(model_path: str, input_texts: list[str] = [], input_file: str = None, output_file: str = None) -> None:
    """Die Hauptfunktion, die entweder CLI-Text oder Datei-I/O ausführt."""
    
    if input_file and output_file:
        # Fall 1: Dateiverarbeitung (wie im Dockerfile CMD gewünscht)
        run_file_prediction(input_file, output_file, model_path)
        
    elif input_texts:
        # Fall 2: Direkte Texteingabe (wie ursprünglich)
        classifier = load_model(model_path)
        preds, probs = predict_texts(classifier, input_texts)
        for line in format_prediction_lines(input_texts, preds, probs):
            print(line)
    
    else:
        # Fall 3: Keine Argumente
        print("Fehler: Bitte geben Sie entweder Text zur Analyse ODER --input und --output Dateipfade an.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Predictor. Analysiert Text oder CSV-Dateien."
    )
    
    # Argumente für das Laden des Modells
    parser.add_argument("--model", default=MODEL_PATH, help="Pfad zur trainierten Pipeline-Datei.")
    
    # NEUE Argumente für Dateiverarbeitung
    parser.add_argument("--input", type=str, default=None, 
                        help="Pfad zur CSV-Datei mit einer 'text'-Spalte.")
    parser.add_argument("--output", type=str, default=None, 
                        help="Pfad, unter dem die Ergebnis-CSV gespeichert werden soll.")
                        
    # Original-Argument für direkte Texteingabe
    parser.add_argument("text", nargs='*', default=[], help="Ein oder mehrere direkt zu bewertende Texte.")
    
    args = parser.parse_args()
    
    # Wir rufen main auf, wobei wir alle möglichen Argumente übergeben
    main(
        model_path=args.model, 
        input_texts=args.text,
        input_file=args.input,
        output_file=args.output
    )