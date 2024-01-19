import logging
import math
import pickle
import re

from collections import defaultdict
from metaphone import doublemetaphone

class Tokenizer:
    def __init__(self, *, regex=r'\b[\w]+\b', oov_token="<OOV>", use_metaphone=False, preprocess_text=None):
        self.word_index = {oov_token: 1}
        self.index_word = {1: oov_token}
        self.regex = regex
        self.use_metaphone = use_metaphone
        self.preprocess_text = preprocess_text if preprocess_text else self.default_preprocessing

    def tokenize(self, text):
        def split_camel_case(word):
            # Insert a space between lower and upper case letters
            word = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', word)
            return word

        # Apply camel case splitting to the original text
        split_words = split_camel_case(text)

        # Then lowercase and tokenize
        processed_words = re.findall(self.regex, split_words.lower())
        return [self.preprocess_word(w) for w in processed_words]

    def fit_on_texts(self, texts):
        unique_words = set(word for text in texts for word in self.tokenize(text))
        # print(unique_words)
        start_index = 2  # Start indexing from 2 because 1 is reserved for OOV
        for i, word in enumerate(sorted(unique_words), start=start_index):
            self.word_index[word] = i
            self.index_word[i] = word

    @staticmethod
    def default_preprocessing(text):
        compound_word_map = {
            r'counter[-\s]?gambit': 'countergambit',
            r'hyper[-\s]?accelerated': 'hyperaccelerated',
            # Add more mappings as needed
        }
        for pattern, replacement in compound_word_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def texts_to_sequences(self, texts):
        def index(word):
            return self.word_index.get(word, 1)  # 1 is the index for OOV

        def tokenize_and_normalize(text):
            normalized_text = self.preprocess_text(text)
            return [index(word) for word in self.tokenize(normalized_text) if len(word) > 1]

        return [tokenize_and_normalize(text) for text in texts]

    def sequences_to_texts(self, sequences):
        return [' '.join(self.index_word.get(i, '') for i in sequence) for sequence in sequences]

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
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
