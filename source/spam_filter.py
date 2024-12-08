# Import required libraries
import joblib  # For loading pre-trained models and vectorizers from disk
from feature_extraction import vectorize_data  # For transforming text data into numerical format (imported but not directly used here)
from data_preprocessing import preprocess_message  # For preprocessing input messages

# Function to test if a given message is spam or not
def test_spam(message, model, vectorizer):
    """
    Classifies a given message as "Spam" or "Ham" using a pre-trained model and vectorizer.

    Steps:
    1. Preprocesses the input message to remove noise and standardize text.
    2. Transforms the preprocessed message into a numerical format using the vectorizer.
    3. Predicts the class (Spam or Ham) using the trained model.
    4. Returns "Spam" if the prediction is 1; otherwise, returns "Ham".

    Args:
        message (str): The input message to be classified.
        model (object): A pre-trained classification model with a `predict` method.
        vectorizer (object): A pre-fitted vectorizer used to convert the message to numerical format.

    Returns:
        str: "Spam" if the message is predicted as spam, "Ham" otherwise.
    """
    # Preprocess the input message
    processed_message = preprocess_message(message)
    
    # Transform the processed message into a numerical format
    message_vector = vectorizer.transform([processed_message])
    
    # Predict the label (1 for spam, 0 for ham)
    prediction = model.predict(message_vector)
    
    # Return the classification result as a string
    return "Spam" if prediction == 1 else "Ham"

# Testing usage with an example:
if __name__ == "__main__":
    # Load the pre-trained model and vectorizer from disk
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    # Test the function with a sample message
    print(test_spam("Free entry to win a prize!", model, vectorizer))
