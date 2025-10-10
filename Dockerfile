# Layer 1: Use an official Python image as a base image
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# 1. Installiere Abhängigkeiten: Dies ist der erste Schritt, um Caching zu maximieren.
# Dieser Layer wird nur neu gebaut, wenn requirements.txt sich ändert.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STAGE 2: Finales Runtime Image (Minimiert die Größe)
FROM python:3.11-slim-bookworm

WORKDIR /app

# Kopiere die Abhängigkeiten aus der Build-Stage
# Wir brauchen keine Build-Tools mehr, nur die installierten Bibliotheken
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 2. Kopiere das GROSSE Modell separat für besseres Caching
# Wenn sich der Code ändert, wird das Modell NICHT erneut kopiert.
# Erstellen Sie den Ordner, falls er nicht existiert, und kopieren Sie das Modell.
RUN mkdir -p models
COPY models/ models/

# 3. Kopiere den Code und die Daten (häufig geändert)
COPY src/ src/
COPY data/ data/

# 4. (Optional) Kopiere die Testdateien, falls Sie Tests im Container ausführen möchten
COPY tests/ tests/

# Layer 7: Specify the command to run on container start
# Führt das Skript zur Vorhersage aus, wie von Ihnen definiert.
CMD ["python", "src/predict.py", "--input", "data/sentiments.csv", "--output", "data/sentiments_out.csv"]