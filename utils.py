import numpy as np

def embed_sequence(seq, embeddings, tfidf_scores):
    if seq:
        weighted_seq = [embeddings[idx] * tfidf_scores.get(idx, 1) for idx in seq]
        return np.mean(weighted_seq, axis=0)
    else:
        return np.zeros(embeddings.shape[1])
