# Demonstração distribuída

Dois processos Ray calculam embeddings; o servidor compara via gRPC.
Esta amostra pequena e aleatória demonstra a integração; a avaliação principal usa pares de teste balanceados.

| Rodada | Classes | Distância | Predição | Acerto |
|---|---|---:|---|---|
| 0 | 9 / 8 | 0.8881 | diferente | True |
| 1 | 0 / 8 | 0.9754 | diferente | True |
| 2 | 8 / 0 | 1.1707 | diferente | True |
| 3 | 9 / 6 | 1.3051 | diferente | True |
| 4 | 7 / 9 | 0.5131 | similar | False |
| 5 | 5 / 1 | 1.0615 | diferente | True |
| 6 | 0 / 8 | 1.2561 | diferente | True |
| 7 | 7 / 9 | 0.7113 | diferente | True |
| 8 | 2 / 4 | 0.7225 | diferente | True |
| 9 | 1 / 8 | 1.4377 | diferente | True |
