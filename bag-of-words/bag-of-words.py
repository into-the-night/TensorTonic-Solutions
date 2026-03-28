import numpy as np
from collections import Counter

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """

    bow = np.zeros(len(vocab), dtype=int)
    count = Counter(tokens)

    for id, word in enumerate(vocab):
        print(word)
        if word in count.keys():
            bow[id] = count[word]            
        
    return bow