# Resultados — Laboratório 5

Execução com os conjuntos completos de treino, validação e teste do CIFAR-10.

- Clientes: 2; épocas executadas: 13 (limite 20); batch: 64.
- Treino: 45000; validação: 5000; teste: 10000 imagens.
- Melhor época pela loss de validação: **8**.
- Teste: **65.61%** de acurácia; loss **0.9902**.
- Atualizações de M2: 9152; tempo incluindo inicialização dos processos: 12.15 min.

| Cliente | Exemplos de teste | Acurácia | Loss |
|---|---:|---:|---:|
| 0 | 5000 | 66.24% | 0.9904 |
| 1 | 5000 | 64.98% | 0.9900 |

![Curvas de treinamento](treinamento.png)

Os três blocos executaram na GPU. Cada cliente tem M1/M3 próprios e usa sua partição de teste. A acurácia agregada é ponderada pelo número de exemplos, não é a avaliação de uma única rede global.

A loss de treino reúne lotes durante atualizações; a validação usa os pesos fixos ao final da época. As curvas não medem exatamente o mesmo estado do modelo. O teste foi avaliado após restaurar o checkpoint conjunto da melhor época, sem orientar a seleção. Pequenas variações numéricas podem ocorrer ao mudar o hardware ou o modo de execução.

`rpc_seconds` mede o tempo observado das duas chamadas, incluindo serviço e serialização; não isola a latência de rede. Os bytes medem mensagens Protobuf e excluem cabeçalhos de transporte. Execução em loopback não caracteriza uma WAN.

Artefatos (caminhos relativos à pasta `lab 5`): `output/config.json`, `output/particoes.json`, `output/historico.csv`, `output/resultados.json`, `output/M2.keras`, `output/cliente_N/M1.keras`, `output/cliente_N/M3.keras` e `output/cliente_N/results.csv`. Os modelos formam um checkpoint de inferência; o estado completo dos otimizadores para retomada não é salvo.

Early stopping coordenado: paciência de 5 épocas, monitorando a menor loss de validação agregada e restaurando o conjunto completo. Dropout: 0.2; pooling adicional: True; Batch Normalization: False; taxa de aprendizado: 0.001.

## Interpretação

Há sinais de sobreajuste após a época selecionada: a loss de treino caiu de 0.7972 para 0.5346, enquanto a de validação passou de 0.9791 para 1.1010. Por isso, a entrega usa o checkpoint escolhido pela validação, preservando a configuração registrada e o limite de 20 épocas desta execução.

Camadas com comportamento distinto entre treino e inferência nesta execução: Dropout. A loss de treino registrada usa regularização ativa e pesos em atualização; por isso, a diferença direta entre treino e validação não isola sobreajuste.

## Tentativa de reduzir o overfitting

Realizamos uma busca com Keras Tuner RandomSearch de 12 configurações, variando Dropout, pooling adicional, Batch Normalization e taxa de aprendizado. As tentativas usaram as mesmas partições e semente, com limite de 20 épocas e paciência 5. A seleção usou somente a loss de validação; o teste foi reservado para o treino final independente da configuração selecionada.

A tentativa não foi suficiente para eliminar os sinais de sobreajuste. O checkpoint entregue corresponde à melhor época pela validação, e não à última época treinada. As configurações e métricas resumidas da busca estão em [busca.json](busca.json). Uma execução por configuração não mede a variabilidade entre sementes nem isola o efeito de cada técnica.
