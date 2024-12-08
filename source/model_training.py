# Import required libraries from scikit-learn
from sklearn.naive_bayes import MultinomialNB  # For the Naive Bayes classification model
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets (imported but unused here)

# Function to train a Naive Bayes model
def train_model(X_train, y_train):
    """
    Trains a Naive Bayes model using the MultinomialNB algorithm.

    Steps:
    1. Initializes the MultinomialNB model.
    2. Fits the model to the provided training data (`X_train`, `y_train`).

    Args:
        X_train (array-like): The feature training data.
        y_train (array-like): The target labels for the training data.

    Returns:
        model (MultinomialNB): A trained Multinomial Naive Bayes model.
    """
    # Initialize the MultinomialNB model
    model = MultinomialNB()
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Return the trained model
    return model
