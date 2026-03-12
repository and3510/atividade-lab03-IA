
# Lab P1 - 03: Implementando o Transformer Decoder "From Scratch"

Este repositório contém a entrega do **Laboratório 3** da disciplina de Tópicos em Inteligência Artificial do **icev**. O objetivo é construir os blocos matemáticos centrais do **Decoder** do modelo Transformer original ("Attention Is All You Need"), operando as matrizes exclusivamente com `numpy`.

## Bibliotecas Utilizadas

* Python 3.x
* numpy

## Como rodar o código

1. Clone este repositório no seu ambiente local.
2. É recomendado usar o seu ambiente virtual já configurado:
```bash
source venv/bin/activate

```


3. Garanta que o `numpy` esteja instalado:
```bash
pip install numpy

```


4. Execute o arquivo principal:
```bash
python script.py

```



## O que o script faz

Diferente do Laboratório 2 que focou no Encoder , este sistema foca na geração fluente de texto, uma palavra por vez. O script executa três tarefas principais:

1. **Máscara Causal (Look-Ahead Masking)**: Implementa uma função que gera uma matriz triangular para impedir que o modelo "olhe para o futuro" durante o processamento, garantindo que a palavra na posição $i$ não atenda à posição $i+1$.


    * A fórmula implementada é: $Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}}+M)V$.




2. **Ponte Encoder-Decoder (Cross-Attention)**: Implementa a integração estrutural onde o estado atual da geração (Decoder) interage com a "memória" lida pelo Encoder através de projeções de matrizes de pesos arbitrárias para $Q$, $K$ e $V$.


3. **Loop de Inferência Auto-Regressivo**: Simula o comportamento real de um modelo de linguagem utilizando um laço `while` que gera tokens iterativamente. O loop utiliza a função `argmax` para selecionar a maior probabilidade e é interrompido imediatamente ao encontrar o token fictício `<EOS>`.



## Validação de Resultados

Ao rodar o script, o console exibirá:

* A matriz de máscara causal $M$ (com valores $-\infty$).


* A prova real de que as probabilidades de palavras futuras no Softmax tornaram-se estritamente $0.0$.


* A dimensão correta da saída do Cross-Attention ($1, 4, 512$).


* O log da frase sendo gerada passo a passo até o token de parada.



## Nota de Crédito

Conforme as regras da disciplina sobre o uso de IA Generativa, registro que utilizei o **Gemini** como colaborador para:

* Auxiliar na transposição da lógica de tensores do PyTorch para a sintaxe pura do `numpy`.
* Brainstorming sobre a estabilidade numérica da função Softmax manual.
* Refinamento da lógica do loop auto-regressivo e estruturação deste README.
A lógica de implementação das funções de máscara e atenção cruzada, bem como a organização dos loops de teste, foram desenvolvidas por mim.

