# Sentiment Analysis Project

## Setup

### Option 1: Python venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### Option 2: Conda
conda create -n sentiment-env python=3.11 -y
conda activate sentiment-env
pip install -r requirements.txt

## Train
python src/train.py --data data/train.csv --out models/sentiment.joblib

## Predict

Run the predictions in the command line. Use any sentence you like.

python src/predict.py "That was the best" "I'm happy" "Whoa! That's terrible" "That's so bad" "That was really anoying"
