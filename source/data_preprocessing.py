import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

def preprocess_message(message):
    message = message.lower()
    message = ''.join([char for char in message if char not in string.punctuation])
    tokens = message.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def preprocess_text_data(file_path):
    nltk.download('stopwords')
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'spam': 1, 'ham': 0})
    data['message'] = data['message'].apply(preprocess_message)
    return data
