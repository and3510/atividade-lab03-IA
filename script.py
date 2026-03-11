import numpy as np

# --- Funções Auxiliares ---

def softmax(x):
    """Calcula o softmax de forma numericamente estável."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

