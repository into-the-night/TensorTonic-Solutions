import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    batch = Q.shape[0]
    seq_len = Q.shape[1]
    d_model = Q.shape[-1]
    d_k = d_model // num_heads
    
    Q_in = Q @ W_q
    K_in = K @ W_k
    V_in = V @ W_v

    # (batch, seq_len, d_model)

    Q_in = Q_in.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_in = K_in.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_in = V_in.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # (batch, num_heads, seq_len, d_k)

    scores = softmax((Q_in @ K_in.transpose(0, 1, 3, 2))/np.sqrt(d_k), axis=-1)

    # (batch, num_heads, seq_len, seq_len)

    attn = scores @ V_in # (batch, num_heads, seq_len, d_k)

    attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    return attn @ W_o
    