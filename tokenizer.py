import logging
import math
import pickle
import re

from collections import defaultdict
from metaphone import doublemetaphone

class Tokenizer:
    def __init__(self, *, regex=r'\b[\w]+\b', oov_token="<OOV>", use_metaphone=False, post_process=None):
        self.word_index = {oov_token: 1}
        self.index_word = {1: oov_token}
        self.regex = regex
        self.use_metaphone = use_metaphone
        self.post_process= post_process if post_process else self.default_post_process

    def tokenize(self, text):
        def split_camel_case(word):
            # Insert a space between lower and upper case letters
            word = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', word)
            return word

        # Apply camel case splitting to the original text
        split_words = split_camel_case(text)

        # Then lowercase and tokenize
        tokens = [tok for tok in re.findall(self.regex, split_words.lower()) if len(tok) > 1]
        tokens = self.post_process(tokens)

        if self.use_metaphone:
            tokens = [code for tok in tokens for code in doublemetaphone(tok) if code]

        return tokens

    def fit_on_texts(self, texts):
        unique_words = set(word for text in texts for word in self.tokenize(text))
        # print(unique_words)
        start_index = 2  # Start indexing from 2 because 1 is reserved for OOV
        for i, word in enumerate(sorted(unique_words), start=start_index):
            self.word_index[word] = i
            self.index_word[i] = word

    @staticmethod
    def default_post_process(tokens):
        compound_word_map = {
            ('counter', 'gambit'): 'countergambit',
            ('hyper', 'accelerated'): 'hyperaccelerated',
            ('ortho', 'schnapp'): 'orthoschnapp',
            # Add more mappings as needed
        }
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i+1])
            if pair in compound_word_map:
                tokens[i] = compound_word_map[pair]
                del tokens[i+1]
            else:
                i += 1
        return tokens

    def texts_to_sequences(self, texts):
        def index(word):
            return self.word_index.get(word, 1)  # 1 is the index for OOV

        def tokenize_and_index(text):
            return [index(word) for word in self.tokenize(text)]

        return [tokenize_and_index(text) for text in texts]

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

