#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
from annoy import AnnoyIndex

def load_embeddings(embeddings_path):
    return np.load(embeddings_path)

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return tokenizer

def preprocess_and_embed(text_data, tokenizer, embeddings, tfidf_scores=None):
    sequences = tokenizer.texts_to_sequences(text_data)

    if tfidf_scores:
        # Use TF-IDF weighted embeddings
        weighted_embeddings = []
        for seq in sequences:
            if seq:
                weighted_seq = [embeddings[idx] * tfidf_scores.get(idx, 1) for idx in seq]
                mean_weighted_seq = np.mean(weighted_seq, axis=0)
                weighted_embeddings.append(mean_weighted_seq)
            else:
                weighted_embeddings.append(np.zeros(embeddings.shape[1]))
        return weighted_embeddings
    else:
        # Use regular embeddings
        return [embeddings[seq[0]] if seq else np.zeros(embeddings.shape[1]) for seq in sequences]

def search(query, embeddings, annoy_index, training_data, tokenizer, top_n=5):
    query_seq = tokenizer.texts_to_sequences([query])
    query_embedding = preprocess_and_embed([query], tokenizer, embeddings, tokenizer.idf)[0]

    similar_indices = annoy_index.get_nns_by_vector(query_embedding, top_n)
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

    annoy_index = AnnoyIndex(embeddings.shape[1], metric='angular')
    annoy_index.load(args.annoy_index)

    with open(args.text_file, 'r', encoding='utf-8') as file:
        text_data = [line.strip() for line in file]

    search(args.query, embeddings, annoy_index, text_data, tokenizer, args.top_n)
