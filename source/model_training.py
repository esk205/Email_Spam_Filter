from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model
