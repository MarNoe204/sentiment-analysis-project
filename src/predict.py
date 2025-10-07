import argparse
from typing import Any
import numpy as np
from numpy.typing import NDArray
from joblib import load


def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    return load(model_path)


# Other functions will go here


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sentiment.joblib")
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    args = parser.parse_args()
    main(model_path=args.model, input_texts=args.text)
