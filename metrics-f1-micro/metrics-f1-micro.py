import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    diff = y_true - y_pred
    tp = np.count_nonzero(diff == 0)
    fp = len(diff) - tp
    return (2 * tp)/(2*tp + fp + fp)
    