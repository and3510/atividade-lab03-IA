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

# ==========================================================
# Tarefa 2: A Ponte Encoder-Decoder (Cross-Attention)
# ==========================================================

def cross_attention(encoder_out, decoder_state):
    """
    Calcula a atenção cruzada entre o estado do Decoder e a saída do Encoder[cite: 32].
    """
    d_model = 512
    
    # Matrizes de pesos arbitrárias para as projeções [cite: 33, 34]
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    
    # Projeta para virar Query (do Decoder), Keys e Values (do Encoder) [cite: 33, 34]
    Q = decoder_state @ W_q
    K = encoder_out @ W_k
    V = encoder_out @ W_v
    
    # Equação do Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k))V [cite: 14, 35]
    # Usando d_k = d_model aqui
    d_k = Q.shape[-1]
    # K.swapaxes(-1, -2) equivale ao K.T para matrizes de múltiplas dimensões (batch)
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_k)
    
    atencao = softmax(scores)
    
    return atencao @ V

# ==========================================================
# Tarefa 3: Simulando o Loop de Inferência Auto-Regressivo
# ==========================================================

# Configurações do Mock [cite: 40, 41]
VOCAB_SIZE = 10000
START_TOKEN = "<START>"
EOS_TOKEN = "<EOS>"

def generate_next_token(current_sequence, encoder_out):
    """Simula a geração de um vetor de probabilidades do tamanho do vocabulário[cite: 40, 41]."""
    # Simulando a saída do Decoder com um vetor aleatório
    logits = np.random.randn(VOCAB_SIZE)
    
    # Força a parada após um certo tamanho para fins de demonstração
    if len(current_sequence) > 6:
        probs = np.zeros(VOCAB_SIZE)
        probs[999] = 1.0 # Vamos fingir que o ID 999 é o <EOS>
        return probs
        
    return softmax(logits)

def loop_inferencia():
    print("--- TAREFA 3: Loop Auto-Regressivo ---")
    
    # Dados fictícios iniciais [cite: 30, 31, 40]
    encoder_output = np.random.randn(1, 10, 512)
    lista_contexto = [START_TOKEN]
    
    # Mapeamento fictício para o token de parada
    id_para_token = {999: EOS_TOKEN}
    
    # O Loop de inferência [cite: 42]
    while True:
        # Chama a função iterativamente [cite: 42]
        probabilidades = generate_next_token(lista_contexto, encoder_output)
        
        # Seleciona a palavra com maior probabilidade (argmax) [cite: 43]
        proximo_id = np.argmax(probabilidades)
        proxima_palavra = id_para_token.get(proximo_id, f"token_{proximo_id}")
        
        # Adiciona a nova palavra na lista [cite: 44]
        lista_contexto.append(proxima_palavra)
        print(f"Token adicionado: {proxima_palavra}")
        
        # Interrupção imediata se for <EOS> [cite: 45]
        if proxima_palavra == EOS_TOKEN:
            break
            
    print("\nFrase Final Gerada:", " ".join(lista_contexto))

if __name__ == "__main__":
    prova_real_tarefa_1()
    
