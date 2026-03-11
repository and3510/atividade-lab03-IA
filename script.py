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

def prova_real_tarefa_1():
    print("--- TAREFA 1: Máscara Causal ---")
    seq_len = 5
    d_k = 64
    
    # Matrizes fictícias Q e K 
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    
    # Cálculo: (Q @ K.T) / sqrt(d_k) 
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # Adicionando a máscara M 
    M = create_causal_mask(seq_len)
    scores_com_mascara = scores + M
    
    # Aplicação do Softmax [cite: 19]
    pesos_atencao = softmax(scores_com_mascara)
    
    print("Máscara Causal M:\n", M)
    print("\nProbabilidades (Softmax):")
    print(np.round(pesos_atencao, 4)) # Prova real: valores futuros devem ser 0.0 [cite: 19]
    print("-" * 30)

