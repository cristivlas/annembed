import logging
import math
import pickle
import re

from collections import defaultdict
from metaphone import doublemetaphone

class Tokenizer:
    def __init__(self, *, regex=r'\b[\w]+\b', oov_token="<OOV>", use_metaphone=False):
        self.word_index = {oov_token: 1}
        self.index_word = {1: oov_token}
        self.regex = regex
        self.use_metaphone = use_metaphone

    def tokenize(self, text):
        return [self.preprocess_word(w) for w in re.findall(self.regex, text.lower())]

    def fit_on_texts(self, texts):
        unique_words = set(word for text in texts for word in self.tokenize(text))
        # print(unique_words)
        start_index = 2  # Start indexing from 2 because 1 is reserved for OOV
        for i, word in enumerate(sorted(unique_words), start=start_index):
            self.word_index[word] = i
            self.index_word[i] = word

    def texts_to_sequences(self, texts):
        return [[self.word_index.get(word, 1) for word in self.tokenize(text)] for text in texts]

    def sequences_to_texts(self, sequences):
        return [' '.join(self.index_word.get(i, '') for i in sequence) for sequence in sequences]

    def save(self, file_path):
        logging.info(f'tokenizer: saving to "{file_path}"')
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        logging.info(f'tokenizer: loading from "{file_path}"')
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def calculate_tfidf(self, texts):
        doc_count = defaultdict(int)
        num_docs = len(texts)

        for text in texts:
            unique_words = set(self.tokenize(text))
            for word in unique_words:
                if word in self.word_index:
                    doc_count[word] += 1

        self.idf = {
            self.word_index[word]: math.log(num_docs / (1 + freq))
            for word, freq in doc_count.items()
        }

    def preprocess_word(self, word):
        return doublemetaphone(word)[0] if self.use_metaphone else word
