import numpy as np
from collections import Counter

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """

    bow = np.zeros(len(vocab), dtype=int)
    count = Counter(tokens)

    vocab_dict = { y: x for x , y in enumerate(vocab)}
    
    for word in count:
        if word in vocab_dict:
            bow[vocab_dict[word]] = count[word]            
        
    return bow