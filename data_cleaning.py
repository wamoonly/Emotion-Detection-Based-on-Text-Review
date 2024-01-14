from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer

# Read the input
df = pd.read_csv('DATAFORCLEANING.csv', encoding='latin-1')

# Drop duplicates
df = df.drop_duplicates(subset=['Reviewer'])

# Keep only the 'English Review' column
df = df[['English Review']]

# Contractions dictionary
contractions_dict = {
    "ain't": "are not", "'s": " is", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he will have", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "I'd": "I would",
    "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have",
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
    "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "that'd": "that would", "that'd've": "that would have", "there'd": "there would", "there'd've": "there would have",
    "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what've": "what have", "when've": "when have", "where'd": "where did", "where've": "where have", "who'll": "who will",
    "who'll've": "who will have", "who've": "who have", "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
    "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
    "you'll've": "you will have", "you're": "you are", "you've": "you have",
}

# Function to handle contractions
def decontracted(phrase):
    for key, value in contractions_dict.items():
        phrase = re.sub(key, value, phrase, flags=re.IGNORECASE)
    return phrase

# Function to remove unwanted symbols
def remove_special_characters(text):
    # Replace multiple consecutive dots with a single dot
    text = re.sub(r'\.{2,}', '.', text)

    # Remove repetitive laughter sounds like "hahaha" or "hehehehe"
    text = re.sub(r'\b(?:haha|hehe)+\b', '', text, flags=re.IGNORECASE)

    # Remove instances of "hmmm" with any number of 'm' characters
    text = re.sub(r'\bh+m+\b', '', text, flags=re.IGNORECASE)

    # Replace consecutive question marks that are not part of a single question
    text = re.sub(r'\?{2,}', '?', text)

    # Replace unwanted symbols (excluding space)
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)

    return cleaned_text

# Function to preprocess text
def preprocess_text(text):
    # Check if the input is a non-null string
    if not isinstance(text, str) or pd.isnull(text):
        return []

    # Handle contractions and specific short forms
    text = decontracted(text)

    # Remove unwanted symbols
    text = remove_special_characters(text)

    # # Tokenize the text
    # tokens = word_tokenize(text)

     # Tokenize the text using RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
    tokens = tokenizer.tokenize(text)

    return tokens

    return tokens

# Apply preprocessing to the 'English Review' column
df['English Review'] = df['English Review'].apply(preprocess_text)

# Save the 'English Review' column as a CSV file
df.to_csv('NasiLemakReviewss.csv', index=False)
