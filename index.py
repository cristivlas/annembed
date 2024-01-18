#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import os
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from annoy import AnnoyIndex
import tensorflow as tf
from tokenizer import Tokenizer

def load_model_and_embeddings(model_path):
    model = tf.keras.models.load_model(model_path)
    embedding_layer = model.get_layer('embedding')
    embeddings = embedding_layer.get_weights()[0]
    return model, embeddings

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def build_annoy_index(embeddings, sequences, tfidf_scores, num_trees):
    embedding_dim = embeddings.shape[1]
    index = AnnoyIndex(embedding_dim, metric='angular')

    for i, seq in enumerate(sequences):
        if seq:
            weighted_embeddings = [embeddings[word_idx] * tfidf_scores.get(word_idx, 1) for word_idx in seq]
            seq_embedding = np.mean(weighted_embeddings, axis=0)
        else:
            seq_embedding = np.zeros(embedding_dim)

        index.add_item(i, seq_embedding)

    index.build(num_trees)
    return index


def main(args):
    # Ensure all output paths exist.
    for path in [args.embeddings_output, args.index_output, args.tokenizer_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load the pre-trained CBOW model and embeddings
    model, embeddings = load_model_and_embeddings(args.model_path)

    # Load and tokenize the training data
    text_data = load_text_file(args.text_file)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    tokenizer.calculate_tfidf(text_data)  # Calculate TF-IDF scores

    # Build the Annoy index
    sequences = tokenizer.texts_to_sequences(text_data)
    annoy_index = build_annoy_index(embeddings, sequences, tokenizer.idf, args.num_trees)

    tokenizer.save(args.tokenizer_output)
    logging.info(f'Saved tokenizer: {args.tokenizer_output}')

    np.save(args.embeddings_output, embeddings)
    logging.info(f'Saved embedding: {args.embeddings_output}')

    annoy_index.save(args.index_output)
    logging.info(f'Saved index: {args.index_output}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Annoy Index for CBOW Model')
    parser.add_argument('--model-path', default='models/cbow', help='Path to the pre-trained CBOW model')
    parser.add_argument('--text-file', required=True, help='Path to the training text data')
    parser.add_argument('--embeddings-output', default='data/embed.npy', help='Path to save model embeddings')
    parser.add_argument('--index-output', default='data/index.ann', help='Path to save the Annoy index')
    parser.add_argument('--tokenizer-output', default='data/tokenizer.pkl', help='Path to save the tokenizer')
    parser.add_argument('--num-trees', type=int, default=32, help='Number of trees for Annoy index')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
