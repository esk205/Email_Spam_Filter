import string  # Imports the `string` module, which provides constants and utilities, such as punctuation characters.

# Function to preprocess the text
def preprocess_text_message(message):
    """
    Preprocesses a given text message by:
    1. Converting all characters to lowercase.
    2. Removing punctuation characters.
    3. Tokenizing the text into words.
    4. Removing stopwords (common words like 'the', 'and') from the tokens.

    Args:
        message (str): The input text message to preprocess.

    Returns:
        str: The cleaned and preprocessed message.
    """
    # Convert the entire message to lowercase for uniformity.
    message = message.lower()
    
    # Remove punctuation characters using a list comprehension.
    # This iterates through each character in the message and excludes those found in `string.punctuation`.
    message = ''.join([char for char in message if char not in string.punctuation])
    
    # Split the message into a list of individual words (tokens).
    tokens = message.split()
    
    # Filter out stopwords using a list comprehension.
    # This excludes any word in the tokens list that matches a stopword from the `stopwords.words('english')` set.
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Join the filtered tokens back into a single string with space-separated words.
    return ' '.join(tokens)

# Apply preprocessing to the messages in the dataset
data['message'] = data['message'].apply(preprocess_text_message)
"""
Applies the `preprocess_text_message` function to the 'message' column of a DataFrame named `data`.
This processes each text entry in the column, returning a new DataFrame with cleaned text.
"""

# Show preprocessed messages
print(data.head())
"""
Prints the first few rows of the `data` DataFrame to display the preprocessed messages.
This is primarily used for verification and debugging.
"""
