# Import necessary libraries
import numpy as np
import nltk

# Uncomment the block below and run once to download the WordNet resource for lemmatization
############## Uncomment this block if not downloaded yet ##############
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
#####################################################################

# Import WordNetLemmatizer from NLTK
from nltk.stem.wordnet import WordNetLemmatizer

# Class for text preprocessing tasks such as tokenization and lemmatization
class Word_Processing:
    def __init__(self) -> None:
        # Initialize a WordNetLemmatizer object
        self.lemmatizer = WordNetLemmatizer()

    # Tokenize a sentence into words
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)
    
    # Lemmatize a word
    def lemma(self, word):
        return self.lemmatizer.lemmatize(word.lower())
    
    # Create a bag of words representation for a given tokenized sentence and list of words
    def bag_words(self, tokenized_sentence, words):
        # Lemmatize each word in the tokenized sentence
        sentence_words = [self.lemma(word) for word in tokenized_sentence]
        # Initialize a numpy array of zeros to represent the bag of words
        bag = np.zeros(len(words), dtype=np.float32)
        # Set the value to 1 for each word in the bag that appears in the sentence
        for idx, w in enumerate(words):
            if w in sentence_words: 
                bag[idx] = 1
            
        return bag
    
if __name__ == "__main__":
    # Create an instance of Word_Processing class
    word_process = Word_Processing()
    # Test lemmatization on a word and print the result
    print(word_process.lemma("Goes"))
