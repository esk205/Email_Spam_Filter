import joblib
from feature_extraction import vectorize_data
from data_preprocessing import preprocess_message

def predict_spam(message, model, vectorizer):
    processed_message = preprocess_message(message)
    message_vector = vectorizer.transform([processed_message])
    prediction = model.predict(message_vector)
    return "Spam" if prediction == 1 else "Ham"

# Example usage:
if __name__ == "__main__":
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print(predict_spam("Free entry to win a prize!", model, vectorizer))
