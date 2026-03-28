import numpy as np
from numpy import linalg as la

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a_norm = la.norm(a)
    if a_norm == 0:
        return 0
    b_norm = la.norm(b)
    if b_norm == 0:
        return 0
    
    return (np.dot(a,b)/(a_norm * b_norm))