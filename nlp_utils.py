import re 
import gensim
import nltk
import en_core_web_sm

nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()
spacy      = en_core_web_sm.load()


def case_normalize(text):
    '''
    Takes in a text string and returns a string with all characters in lower case.
    '''
    return text.lower()

def remove_punctuation(text):
    '''
    Takes in a text string and returns a string with all the punctuation removed.
    '''
    return re.sub('[^a-zA-Z0-9]', ' ', text)

def tokenize(text):
    '''
    Takes in a text string and returns a list where each item corresponds to a token.
    '''
    return gensim.utils.tokenize(text)

def lemmatize_nltk(text):
    '''
    Takes in a list of tokens, lemmatizes each token according to nltk.stem.WordNetLemmatizer docs and returns text.
    '''
    return ' '.join([lemmatizer.lemmatize(word) for word in text])

def lemmatize_spacy(text):
    '''
    Takes in text, lemmatizes each token according to en_core_web_sm.load docs and returns a list.
    '''
    return [token.lemma_ for token in spacy(text)]

def clean_text(text):
    '''
    Takes in a raw text document and performs the following steps in order:
    - case normalization
    - punctuation removal
    - tokenization
    - lemmatization (nltk)
    - lemmatization (spacy)
    Then returns a string containing the processed text
    '''
    text = case_normalize(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = lemmatize_nltk(text)
    text = lemmatize_spacy(text)
    text = ' '.join(text)
    return text 