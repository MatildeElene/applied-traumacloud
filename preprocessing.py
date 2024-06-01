import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remove HTML entities
    text = re.sub(r'&\w+;', '', text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Remove ellipsis
    text = re.sub(r"\.{3}", " ", text)
    # Replace consecutive digits with spaces
    text = re.sub(r'\d+', ' ', text)
    # Replace "--" and "”" with spaces
    text = text.replace("--", " ")
    text = text.replace("”", " ")
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Remove single letters
    text = re.sub(r'\b\w\b', '', text)
    return text

def tokenize_and_lemmatize(text):
    stop_words = set(stopwords.words('english'))
    text = word_tokenize(text)
    # Remove single letters
    text = [word for word in text if len(word) > 1]
    text = [word for word in text if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word, pos='v') for word in text]
    return text
