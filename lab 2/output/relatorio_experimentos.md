# Relatório de Experimentos Federados

## Configuração

- Dataset: `load_digits`, convertido em classificação binária (pares = 0, ímpares = 1).
- Clientes por cenário: 5.
- Cenários: 1, 2 e 3 clientes bizantinos para cada tipo de ataque.
- Critério de aceitação: acurácia no teste maior que `0.5` (chute aleatório).
- Agregação: média dos pesos dos modelos aceitos.

## Resultados

| Tipo de ataque | Bizantinos | Aceitos | Rejeitados | Com detecção | Sem detecção |
|---|---:|---:|---:|---:|---:|
| ruido em todas as features | 1 | 4 | 1 | 0.8972 | 0.8917 |
| ruido em todas as features | 2 | 3 | 2 | 0.9056 | 0.9000 |
| ruido em todas as features | 3 | 2 | 3 | 0.8972 | 0.8972 |
| ruido em uma feature | 1 | 5 | 0 | 0.9000 | 0.9000 |
| ruido em uma feature | 2 | 5 | 0 | 0.9000 | 0.9000 |
| ruido em uma feature | 3 | 5 | 0 | 0.9028 | 0.9028 |
| rotulos invertidos | 1 | 4 | 1 | 0.8972 | 0.8972 |
| rotulos invertidos | 2 | 3 | 2 | 0.9056 | 0.6944 |
| rotulos invertidos | 3 | 2 | 3 | 0.8972 | 0.2556 |

## Observação

A comparação mostra o desempenho do modelo final após a filtragem dos updates em relação à agregação de todos os modelos recebidos.
