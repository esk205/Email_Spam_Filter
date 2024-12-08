# Import required library
from sklearn.feature_extraction.text import CountVectorizer  
# Provides the CountVectorizer class, which converts text data into a bag-of-words representation.

# Function to vectorize training and testing datasets
def vectorize_data(X_train, X_test):
    """
    Vectorizes the training and testing datasets using the bag-of-words model.

    Steps:
    1. Initializes a CountVectorizer object to transform text into a sparse matrix of token counts.
    2. Fits the vectorizer to the training data (`X_train`) and transforms it into a document-term matrix.
    3. Transforms the testing data (`X_test`) using the same vectorizer, ensuring consistency in token representation.

    Args:
        X_train (list or array-like): The training dataset containing raw text messages.
        X_test (list or array-like): The testing dataset containing raw text messages.

    Returns:
        tuple:
            - X_train_counts (sparse matrix): Bag-of-words representation of the training data.
            - X_test_counts (sparse matrix): Bag-of-words representation of the testing data.
            - vectorizer (CountVectorizer): The fitted CountVectorizer object for further use or inspection.
    """
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()
    
    # Fit the vectorizer to the training data and transform it to a sparse matrix
    X_train_counts = vectorizer.fit_transform(X_train)
    
    # Transform the testing data using the fitted vectorizer
    X_test_counts = vectorizer.transform(X_test)
    
    # Return the transformed datasets and the vectorizer object
    return X_train_counts, X_test_counts, vectorizer
