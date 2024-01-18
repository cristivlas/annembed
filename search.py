#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import tokenizer
from annoy import AnnoyIndex

def load_embeddings(embeddings_path):
    # Load pre-trained word embeddings from a numpy file
    return np.load(embeddings_path)

def load_tokenizer(tokenizer_path):
    # Load the Tokenizer from a pickled file
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return tokenizer

def preprocess_and_embed(text_data, tokenizer, embeddings):
    # Preprocess the entire training data and compute embeddings
    sequences = tokenizer.texts_to_sequences(text_data)
    return [embeddings[seq[0]] if seq else np.zeros(embeddings.shape[1]) for seq in sequences]

def search(query, embeddings, annoy_index, training_data, tokenizer, top_n=5):
    # Tokenize the query phrase and convert it into an embedding
    query_seq = tokenizer.texts_to_sequences([query])
    query_embedding = np.mean([embeddings[i] for i in query_seq[0]], axis=0) if query_seq[0] else np.zeros(embeddings.shape[1])

    # Perform similarity search using Annoy
    similar_indices = annoy_index.get_nns_by_vector(query_embedding, top_n)

    # Retrieve and print the top N similar phrases from the training data
    print("Top {} similar phrases for query '{}':".format(top_n, query))
    for idx in similar_indices:
        print(training_data[idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Similarity Search')
    parser.add_argument('--query', required=True, help='Query phrase')
    parser.add_argument('--embeddings', default='data/embed.npy', help='Path to pre-trained word embeddings (numpy file)')
    parser.add_argument('--annoy-index', default='data/index.ann', help='Path to Annoy index')
    parser.add_argument('--tokenizer', default='data/tokenizer.pkl', help='Path to pickled Tokenizer')
    parser.add_argument('--text-file', required=True, help='Path to text data')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top matches to retrieve')
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings)
    tokenizer = load_tokenizer(args.tokenizer)

    # Load the Annoy index
    annoy_index = AnnoyIndex(embeddings.shape[1], metric='angular')
    annoy_index.load(args.annoy_index)

    # Load and preprocess the training data, then compute embeddings
    with open(args.text_file, 'r', encoding='utf-8') as file:
        text_data = [line.strip() for line in file]

    embeddings_data = preprocess_and_embed(text_data, tokenizer, embeddings)
    search(args.query, embeddings_data, annoy_index, text_data, tokenizer, args.top_n)