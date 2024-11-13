import string

# Function to preprocess the text
def preprocess_text_message(message):
    # Convert to lowercase
    message = message.lower()
    # Remove punctuation
    message = ''.join([char for char in message if char not in string.punctuation])
    # Tokenize and remove stopwords
    tokens = message.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing to the messages
data['message'] = data['message'].apply(preprocess_text_message)

# Show preprocessed messages
print(data.head())
