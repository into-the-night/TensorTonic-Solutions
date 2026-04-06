import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    pos = np.arange(seq_length).reshape(-1,1)
    dims = np.arange(d_model)
    pos = pos/np.pow(10000, 2*dims/d_model)
    pos = np.where( dims % 2 == 0, np.sin(pos), np.cos(pos))

    return pos