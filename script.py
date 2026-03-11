import numpy as np

# --- Funções Auxiliares ---

def softmax(x):
    """Calcula o softmax de forma numericamente estável."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ==========================================================
# Tarefa 1: Implementando a Máscara Causal (Look-Ahead Mask)
# ==========================================================

def create_causal_mask(seq_len):
    """
    Cria uma matriz quadrada onde a parte triangular inferior (e diagonal) 
    contém 0 e a superior contém -infinito[cite: 16, 17].
    """
    # Cria uma matriz cheia de zeros
    mask = np.zeros((seq_len, seq_len))
    
    # Obtém os índices do triângulo superior (acima da diagonal principal)
    indices_superior = np.triu_indices(seq_len, k=1)
    
    # Preenche a parte superior com infinito negativo [cite: 17]
    mask[indices_superior] = -np.inf
    
    return mask
