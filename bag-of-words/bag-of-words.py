import numpy as np
from collections import Counter

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """

    bow = np.zeros(len(vocab), dtype=int)
    count = Counter(tokens)

    for word in tokens:
        if word in vocab:
            bow[vocab.index(word)] = count[word]            
        
    return bow