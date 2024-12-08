# Import required metrics from scikit-learn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
# Provides functions for evaluating model performance with metrics like accuracy, confusion matrix, and detailed classification reports.

# Function to evaluate a model's performance
def evaluation_model(model, X_test, y_test):
    """
    Evaluates a given model's performance on the test data using common classification metrics.

    Steps:
    1. Predicts the target values for the `X_test` dataset.
    2. Calculates the accuracy score comparing actual and predicted values.
    3. Computes the confusion matrix to show true/false positives and negatives.
    4. Generates a detailed classification report including precision, recall, F1-score, and support.

    Args:
        model (object): A trained machine learning model with a `predict` method.
        X_test (array-like): The feature data used for testing the model.
        y_test (array-like): The true target values for the test data.

    Prints:
        - Accuracy score of the model.
        - Confusion matrix showing counts of true positives, false positives, true negatives, and false negatives.
        - Classification report detailing precision, recall, F1-score, and support for each class.
    """
    # Predict the target values for the test data
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Create the classification report
    report = classification_report(y_test, y_pred)
    
    # Print the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{report}")
