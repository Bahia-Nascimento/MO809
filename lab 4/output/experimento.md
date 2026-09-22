# Demonstração distribuída

Dois processos Ray calculam embeddings; o servidor compara via gRPC.
Esta amostra pequena e aleatória demonstra a integração; a avaliação principal usa pares de teste balanceados.

| Rodada | Classes | Distância | Predição | Acerto |
|---|---|---:|---|---|
| 0 | 9 / 8 | 0.8113 | diferente | True |
| 1 | 0 / 8 | 0.8563 | diferente | True |
| 2 | 8 / 0 | 0.9358 | diferente | True |
| 3 | 9 / 6 | 1.3604 | diferente | True |
| 4 | 7 / 9 | 0.7141 | diferente | True |
| 5 | 5 / 1 | 1.2115 | diferente | True |
| 6 | 0 / 8 | 1.1858 | diferente | True |
| 7 | 7 / 9 | 0.7758 | diferente | True |
| 8 | 2 / 4 | 0.6758 | diferente | True |
| 9 | 1 / 8 | 1.4023 | diferente | True |
