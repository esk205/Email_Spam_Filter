from sklearn.feature_extraction.text import CountVectorizer

def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    return X_train_counts, X_test_counts, vectorizer
