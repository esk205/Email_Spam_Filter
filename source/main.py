from sklearn.model_selection import train_test_split
import joblib

from data_preprocessing import preprocess_data
from feature_extraction import vectorize_data
from model_training import train_model
from model_evaluation import evaluate_model
from spam_filter import predict_spam

def main():
    # 1. Load and preprocess the dataset
    data = preprocess_data('data/spam.csv')
    
    # 2. Split the data into training and testing sets
    X = data['message']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Extract features using CountVectorizer
    X_train_counts, X_test_counts, vectorizer = vectorize_data(X_train, X_test)
    
    # 4. Train the Naive Bayes model
    model = train_model(X_train_counts, y_train)
    
    # 5. Evaluate the model
    evaluate_model(model, X_test_counts, y_test)
    
    # 6. Save the model and vectorizer for future use
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    # 7. Test the spam filter with new messages
    print(predict_spam("Free entry to win a prize!", model, vectorizer))
    print(predict_spam("Hey, are you free this evening to catch up?", model, vectorizer))

if __name__ == "__main__":
    main()
