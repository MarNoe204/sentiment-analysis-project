import pytest
# We import the test helper function that uses the MockClassifier
from src.predict import run_mock_prediction 

# --- Parametrization (Bonus Challenge) ---

# This list contains 23 film comments from the sentiment.csv file, 
# converted from (text, 1/0) to (text, 'positive'/'negative').
@pytest.mark.parametrize("input_text, expected_sentiment", [
    ("I loved this film", "positive"),
    ("Terrible acting and a weak plot", "negative"),
    ("Surprisingly good", "positive"),
    ("That was bad", "negative"),
    ("Not my cup of tea", "negative"),
    ("Absolutely brilliant from start to finish", "positive"),
    ("Utter rubbish, I almost walked out", "negative"),
    ("Decent story and lovely soundtrack", "positive"),
    ("Boring and far too long", "negative"),
    ("The cast were fantastic", "positive"),
    ("The dialogue was cringeworthy", "negative"),
    ("Exceeded my expectations", "positive"),
    ("I would not recommend it", "negative"),
    ("A charming, feel-good watch", "positive"),
    ("Confusing plot and poor pacing", "negative"),
    ("Kept me on the edge of my seat", "positive"),
    ("Predictable with no surprises", "negative"),
    ("Beautifully shot and well acted", "positive"),
    ("The jokes fell flat", "negative"),
    ("An instant favourite", "positive"),
    ("Disappointing ending", "negative"),
    ("A true masterpiece of cinema", "positive"),
    ("I wanted my money back", "negative"),
])
def test_predict_obvious_sentiments(input_text: str, expected_sentiment: str):
    """
    Tests the prediction pipeline using a mock classifier to ensure basic
    positive and negative inputs yield the correct string labels ('positive'/'negative').
    This fulfills the 'Sanity Check' requirement using parametrization.
    """
    
    # The helper function run_mock_prediction requires a list of texts
    input_list = [input_text]
    
    # Get prediction labels
    predictions = run_mock_prediction(input_list)
    
    # Assert: The first (and only) prediction must match the expected sentiment string
    assert predictions[0] == expected_sentiment, \
        (f"Test failed for '{input_text}'. "
         f"Expected: '{expected_sentiment}', Received: '{predictions[0]}'")

# --- Format Test (Similar to checking DataFrame shape/columns) ---

def test_prediction_output_list_format():
    """
    Ensures the run_mock_prediction function always returns a list of strings
    with the same length as the input list, and that all returned strings are valid labels.
    """
    sample_texts = ["Test one", "Test two", "Test three"]
    predictions = run_mock_prediction(sample_texts)
    
    # Assert 1: Check the length (Must be 3 predictions for 3 inputs)
    assert len(predictions) == len(sample_texts), \
        "The number of predictions does not match the input length."
    
    # Assert 2: Check the type (All elements must be strings)
    assert all(isinstance(p, str) for p in predictions), \
        "All prediction elements must be strings."
    
    # Assert 3: Check for valid labels
    valid_labels = {'positive', 'negative'}
    assert all(p in valid_labels for p in predictions), \
        "Prediction contained an invalid sentiment label."
