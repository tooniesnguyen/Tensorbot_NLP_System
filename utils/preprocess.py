import numpy as np
import nltk



############## Nếu down r thì comment block này lại ############## 
nltk.download('omw-1.4')
#####################################################################
       
from nltk.stem.wordnet import WordNetLemmatizer

class Word_Processing:
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)
    
    def lemma(self, word):
        return self.lemmatizer.lemmatize(word.lower())
    
    def bag_words(self, tokenized_sentence, words):
        sentence_words = [self.lemma(word) for word in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words: 
                bag[idx] = 1
        return bag
    
if __name__ == "__main__":
    word_process = Word_Processing()
    print(word_process.lemma("Goes"))
