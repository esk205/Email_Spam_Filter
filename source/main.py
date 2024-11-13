from sklearn.model_selection import train_test_split
import joblib

from data_preprocessing import preprocess_text_data
from feature_extraction import vectorize_data
from model_training import train_model
from model_evaluation import evaluation_model
from spam_filter import test_spam

def main():
    # 1. Load and preprocess the dataset
    data = preprocess_text_data('data/spam.csv')
    
    # 2. Split the data into training and testing sets
    X = data['message']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Extract features using CountVectorizer
    X_train_counts, X_test_counts, vectorizer = vectorize_data(X_train, X_test)
    
    # 4. Train the Naive Bayes model
    model = train_model(X_train_counts, y_train)
    
    # 5. Evaluate the model
    evaluation_model(model, X_test_counts, y_test)
    
    # 6. Save the model and vectorizer for future use
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    user_input = input("Please enter an email to test or hit 0 for an example: ")
    print("You entered:", user_input)

    if user_input == "0":
        print("You entered zero. So a test example will be used")

        # 7. Test the spam filter with new messages
        print(test_spam("Free entry to win a grand prize!", model, vectorizer)) # if the test replies with spam then its spam

        # if the test replies with ham then it's not spam
        print(test_spam("Hey, are you free this evening for a coffee to catch up?", model, vectorizer))

        print("So the first email: Free entry to win a grand prize! Turned out to be spam")
        print("So the second email: Hey, are you free this evening for a coffee to catch up? Turned out to NOT be spam thus labelled ham")

    else:
        print(test_spam(user_input, model, vectorizer))
        print("If the test spam you inputted turned out to be labelled as 'ham' then its NOT spam. If it got labelled as spam then it IS spam")

if __name__ == "__main__":
    main()
