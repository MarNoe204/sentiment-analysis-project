# Dockerfile: Sentiment Analysis Application

# Basis-Image: Python 3.11 Slim
FROM python:3.11-slim-bookworm

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die requirements.txt und installiere Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den gesamten Quellcode (src/, data/, etc.)
# Das trainierte Modell muss im lokalen models/ Ordner existieren, bevor gebaut wird!
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY tests/ tests/

# [NEU] Setze den ENTRYPOINT auf Python. Alle nachfolgenden Argumente 
# werden als Argumente an Python übergeben.
# ENTRYPOINT wird immer ausgeführt, CMD liefert die Standardargumente.
ENTRYPOINT ["python", "src/predict.py"]

# CMD wird jetzt NUR das Standard-Argument (den Text) an das ENTRYPOINT-Skript übergeben.
# Wir lassen CMD leer, da der Benutzer den Text als Argument übergeben soll.
CMD [] 
