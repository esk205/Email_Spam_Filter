# Import required libraries
import pandas as pd  # For data manipulation and handling CSV files
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # For access to stopword lists
import string  # Provides constants and utilities, such as punctuation characters

# Function to preprocess a single message
def preprocess_message(message):
    """
    Cleans and preprocesses a single text message by:
    1. Converting the message to lowercase.
    2. Removing punctuation characters.
    3. Tokenizing the message into individual words (tokens).
    4. Removing common stopwords (e.g., 'and', 'the').

    Args:
        message (str): The input text message to be preprocessed.

    Returns:
        str: The preprocessed message as a single string.
    """
    # Convert the message to lowercase for uniformity
    message = message.lower()
    
    # Remove punctuation characters using a list comprehension
    message = ''.join([char for char in message if char not in string.punctuation])
    
    # Split the message into individual tokens (words)
    tokens = message.split()
    
    # Remove stopwords from the tokens list
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Combine the tokens back into a single string with words separated by spaces
    return ' '.join(tokens)

# Function to preprocess an entire dataset
def preprocess_text_data(file_path):
    """
    Loads, preprocesses, and cleans a text dataset from a CSV file.

    Steps:
    1. Downloads the stopwords required by the `nltk` library.
    2. Reads the CSV file located at `file_path` using `pandas.read_csv`.
    3. Extracts and renames specific columns ('v1' as 'label' and 'v2' as 'message').
    4. Converts message labels ('spam', 'ham') to numerical format (1 for 'spam', 0 for 'ham').
    5. Applies text preprocessing to each message in the dataset.

    Args:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: A cleaned and preprocessed DataFrame with two columns:
            - 'label': Binary indicator for spam (1) or ham (0).
            - 'message': Preprocessed text messages.
    """
    # Download the stopwords from nltk
    nltk.download('stopwords')
    
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path, encoding='latin-1')
    
    # Select only the relevant columns: 'v1' (label) and 'v2' (message)
    data = data[['v1', 'v2']]
    
    # Rename the columns for better clarity
    data.columns = ['label', 'message']
    
    # Map the 'label' column to binary values: 'spam' -> 1, 'ham' -> 0
    data['label'] = data['label'].map({'spam': 1, 'ham': 0})
    
    # Apply the `preprocess_message` function to clean the 'message' column
    data['message'] = data['message'].apply(preprocess_message)
    
    # Return the preprocessed DataFrame
    return data
