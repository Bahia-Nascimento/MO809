# Relatório de Experimentos Cliente-Servidor

## Configuração

- Dataset: `load_digits`, convertido em classificação binária (pares = 0, ímpares = 1).
- Clientes por cenário: 5.
- Cenários: 1, 2 e 3 clientes bizantinos para cada tipo de ataque.
- Critério de aceitação: acurácia no teste maior que `0.5` (chute aleatório).
- Treinamento final realizado no servidor com os dados recebidos.

## Resultados

| Tipo de ataque | Bizantinos | Aceitos | Rejeitados | Acurácias individuais (clientes 0-4) | Com detecção | Sem detecção |
|---|---:|---:|---:|---|---:|---:|
| ruido em todas as features | 1 | 4 | 1 | 0.4528;0.9583;0.9722;0.9611;0.9611 | 0.9861 | 0.9833 |
| ruido em todas as features | 2 | 4 | 1 | 0.4528;0.5389;0.9722;0.9611;0.9611 | 0.9917 | 0.9889 |
| ruido em todas as features | 3 | 3 | 2 | 0.4528;0.5389;0.4278;0.9611;0.9611 | 0.9806 | 0.9778 |
| ruido nas ultimas 8 features | 1 | 5 | 0 | 0.9389;0.9583;0.9722;0.9611;0.9611 | 0.9889 | 0.9889 |
| ruido nas ultimas 8 features | 2 | 5 | 0 | 0.9389;0.9528;0.9722;0.9611;0.9611 | 0.9917 | 0.9917 |
| ruido nas ultimas 8 features | 3 | 5 | 0 | 0.9389;0.9528;0.9583;0.9611;0.9611 | 0.9833 | 0.9833 |
| rotulos invertidos | 1 | 4 | 1 | 0.0528;0.9583;0.9722;0.9611;0.9611 | 0.9861 | 0.9750 |
| rotulos invertidos | 2 | 3 | 2 | 0.0528;0.0417;0.9722;0.9611;0.9611 | 0.9917 | 0.8000 |
| rotulos invertidos | 3 | 2 | 3 | 0.0528;0.0417;0.0278;0.9611;0.9611 | 0.9778 | 0.2556 |

## Observação

A comparação mostra o desempenho do modelo final após a filtragem dos clientes em relação ao treinamento usando todos os dados recebidos.
