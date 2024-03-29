#!/usr/bin/env python3
import argparse
import numpy as np
import logging
import os

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Embedding, Dense, MultiHeadAttention, GlobalAveragePooling1D, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tokenizer import Tokenizer

# https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1
# https://medium.com/analytics-vidhya/understanding-positional-encoding-in-transformers-def92aca1dfe
# https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3

def get_positional_encoding(max_seq_len, embed_dim):
    positional_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / embed_dim) for j in range(embed_dim)]
        for pos in range(max_seq_len)])
    positional_enc[:, 0::2] = np.sin(positional_enc[:, 0::2])  # apply sin to even indices
    positional_enc[:, 1::2] = np.cos(positional_enc[:, 1::2])  # apply cos to odd indices
    return positional_enc

class CBOW(Model):
    """ Continuous Bag of Words with Positional Encoding. """
    def __init__(self, vocab_size, embedding_dim, window_size, num_heads=4):
        super(CBOW, self).__init__()
        assert num_heads > 0
        assert embedding_dim % num_heads == 0, "num_heads must be a positive divisor of embedding_dim"

        self.embedding = Embedding(vocab_size, embedding_dim, input_length=2*window_size)
        self.positional_encoding = get_positional_encoding(2*window_size, embedding_dim)
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.global_average_pooling = GlobalAveragePooling1D()
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, context):
        context_emb = self.embedding(context)  # Embedding layer
        context_emb += self.positional_encoding[:context.shape[1], :]  # Add positional encoding
        attention_output = self.multi_head_attention(context_emb, context_emb)  # Transformer block
        proj_input = self.layer_norm(context_emb + attention_output)
        pooled_output = self.global_average_pooling(proj_input)  # Global average pooling
        return self.dense(pooled_output)  # Final dense layer

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def generate_cbow_pairs(sequences, vocab_size, window_size):
    context_windows, labels = [], []
    for sequence in sequences:
        length = len(sequence)
        for i, word in enumerate(sequence):
            context = [0] * (window_size * 2)
            label = word
            k = 0
            for j in range(i - window_size, i + window_size + 1):
                if j != i and 0 <= j < length:
                    context[k] = sequence[j]
                    k += 1
            context_windows.append(context)
            labels.append(label)
    return np.array(context_windows), tf.keras.utils.to_categorical(labels, num_classes=vocab_size)

def configure_logging(args):
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=args.log_filename,
            filemode='a')
    tf_logger = tf.get_logger()
    tf_logger.handlers = []  # Remove the default TensorFlow handlers
    tf_logger.addHandler(logging.StreamHandler())  # Add the default Python logging handler

def main(args):
    # Ensure all output paths exist.
    for path in [args.model_path, args.tokenizer_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    text_data = load_text_file(args.text_file)

    tokenizer = Tokenizer(use_metaphone=args.use_metaphone)
    tokenizer.fit_on_texts(text_data)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because index 0 is reserved
    logging.info(f'vocab_size={vocab_size}')

    tokenizer.calculate_tfidf(text_data)  # Calculate TF-IDF scores
    tokenizer.save(args.tokenizer_output)
    logging.info(f'Saved tokenizer: {args.tokenizer_output}')

    sequences = tokenizer.texts_to_sequences(text_data)

    context_pairs, target_labels = generate_cbow_pairs(sequences, vocab_size, args.window_size)

    if os.path.exists(args.model_path):
        # Load existing model.
        model = tf.keras.models.load_model(args.model_path)
        logging.info(f'Loaded model from: {args.model_path}')
        model.summary(line_length=120)
    else:
        # Create and compile the CBOW model.
        model = CBOW(vocab_size, args.embedding_dim, args.window_size)
        model.compile(optimizer=Adam(amsgrad=True), loss='categorical_crossentropy')

    if args.epochs > 0:
        callbacks = []
        if args.checkpoint:
            checkpoint = ModelCheckpoint(args.model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks.append(checkpoint)

        # Train the model
        model.fit(context_pairs, target_labels, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)

        if not args.checkpoint:
            model.save(args.model_path)

if __name__ == "__main__":
    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        ...
    parser = argparse.ArgumentParser(description='Train CBOW Model', formatter_class=Formatter)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=256)
    parser.add_argument('--checkpoint', action='store_true', help='Use checkpoint callback')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Dimension of embedding vector')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--log-filename', default='log.txt', help='Log filename')
    parser.add_argument('--model-path', default='models/cbow', help='Path to save the trained model')
    parser.add_argument('--text-file', required=True, help='Path to the text file for training data')
    parser.add_argument('--tokenizer-output', default='data/tokenizer.pkl', help='Path to save the tokenizer')
    parser.add_argument('--use-metaphone', action='store_true')
    parser.add_argument('--window-size', type=int, default=3,
                        help='Number of neighboring words considered around a target word')
    args = parser.parse_args()

    try:
        configure_logging(args)
        main(args)
    except KeyboardInterrupt:
        print()
