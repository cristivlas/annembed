#!/usr/bin/env python3
import argparse
import numpy as np
import pickle

import os
import sys
# Hack: Ensure tokenizer module is found at unpickle time.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import tokenizer

from annoy import AnnoyIndex
from utils import embed_sequence

def load_embeddings(embeddings_path):
    return np.load(embeddings_path)

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return tokenizer

def preprocess_and_embed(text_data, tokenizer, embeddings):
    tfidf_scores = tokenizer.idf
    assert tfidf_scores

    sequences = tokenizer.texts_to_sequences(text_data)
    return [embed_sequence(s, embeddings, tfidf_scores) for s in sequences]

class Index:
    def __init__(self, index_dir):
        self.embeddings = load_embeddings(os.path.join(index_dir, 'data', 'embed.npy'))
        self.index = AnnoyIndex(self.embeddings.shape[1], metric='angular')
        self.index.load(os.path.join(index_dir, 'data', 'index.ann'))
        self.tokenizer = load_tokenizer(os.path.join(index_dir, 'data', 'tokenizer.pkl'))

    def search(self, query, *, max_distance=None, top_n=5, min_nodes=50):
        query_embed = preprocess_and_embed([query], self.tokenizer, self.embeddings)[0]
        k = max(min_nodes, top_n * self.index.get_n_trees())
        similar_indices = self.index.get_nns_by_vector(query_embed, n=top_n, search_k=k, include_distances=True)
        assert similar_indices is not None
        return [(i,d) for i,d in zip(*similar_indices) if max_distance is None or d < max_distance]

def search(query, embeddings, annoy_index, data, tokenizer, top_n=5):
    query_seq = tokenizer.texts_to_sequences([query])
    query_embedding = preprocess_and_embed([query], tokenizer, embeddings)[0]

    similar_indices = annoy_index.get_nns_by_vector(query_embedding, top_n, include_distances=True)
    print("Top {} similar phrases for query '{}':".format(top_n, query))
    for idx, distance in zip(*similar_indices):
        print(f'{data[idx]}, d={distance}')

if __name__ == "__main__":
    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        ...
    parser = argparse.ArgumentParser(description='Perform Similarity Search', formatter_class=Formatter)
    parser.add_argument('--query', required=True, help='Query phrase')
    parser.add_argument('--embeddings', default='data/embed.npy', help='Path to pre-trained word embeddings (numpy file)')
    parser.add_argument('--annoy-index', default='data/index.ann', help='Path to Annoy index')
    parser.add_argument('--tokenizer', default='data/tokenizer.pkl', help='Path to pickled Tokenizer')
    parser.add_argument('--text-file', required=True, help='Path to text data')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top matches to retrieve')
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings)
    tokenizer = load_tokenizer(args.tokenizer)

    annoy_index = AnnoyIndex(embeddings.shape[1], metric='angular')
    annoy_index.load(args.annoy_index)

    with open(args.text_file, 'r', encoding='utf-8') as file:
        text_data = [line.strip() for line in file]

    search(args.query, embeddings, annoy_index, text_data, tokenizer, args.top_n)
