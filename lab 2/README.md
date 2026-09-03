# Laboratório 2 - Clientes Bizantinos

Este projeto simula cinco clientes em uma arquitetura cliente-servidor. Os clientes recebem partições dos dados, aplicam comportamentos normais ou bizantinos e enviam as amostras ao servidor por gRPC. O servidor identifica clientes suspeitos e treina os modelos finais com os dados recebidos.

## Como executar

O ambiente virtual do projeto já contém as dependências necessárias. No PowerShell, a partir da pasta `lab 2`:

```powershell
..\venv\Scripts\Activate.ps1
python experimento.py
```

O script inicia o servidor, inicializa um cluster Ray local, cria cinco actors para os clientes e executa os nove cenários:

- três tipos de comportamento bizantino;
- um, dois e três clientes bizantinos em cada tipo.

Os resultados são gravados em:

- `output/relatorio_experimentos.csv`
- `output/relatorio_experimentos.md`

Para executar um cliente individualmente, inicie primeiro o servidor com `python mlservidor.py` e, em outro terminal, use:

```powershell
python mlcliente.py <client_id> <behavior_id>
```

`client_id` varia de `0` a `4`. `behavior_id` possui os seguintes valores:

| ID | Comportamento |
|---:|---|
| 0 | Cliente normal |
| 1 | Substitui todas as features por valores aleatórios |
| 2 | Substitui as últimas 8 features por valores aleatórios |
| 3 | Inverte os rótulos binários |

Exemplo:

```powershell
python mlcliente.py 4 3
```

## Organização do sistema

1. O servidor carrega o dataset e separa 80% para treino e 20% para teste.
2. Os dados de treino são distribuídos em cinco partições disjuntas, identificadas pelo `client_id`.
3. Cada cliente recebe somente sua partição por meio do RPC `GetTrainingData`.
4. Cada cliente aplica seu comportamento e envia os dados no RPC `SubmitDataUpdate`.
5. O servidor espera os cinco envios antes de processá-los.
6. O servidor treina um modelo separado com os dados de cada cliente e avalia cada modelo no teste reservado.
7. O servidor aceita clientes cujo modelo supera a acurácia de um chute aleatório.
8. O servidor treina um modelo final com todos os dados e outro somente com os dados dos clientes aceitos.

## Decisões tomadas

### Comunicação via gRPC

O Lab 1 já fornecia a comunicação gRPC e a estrutura protobuf. Mantivemos essa base e acrescentamos mensagens para solicitar partições e enviar os dados alterados pelos clientes.

### Cinco processos de cliente com Ray

Para paralelizar chamadas RPC para cinco clientes, criamos `ClienteActor` com `@ray.remote`. Cada actor representa um cliente independente e executa seu treinamento em um worker separado.

### Dataset `load_digits`

O Iris era pequeno para observar o efeito dos ataques. O `load_digits` possui 1.797 amostras e 64 features. Os rótulos foram convertidos em um problema binário:

```text
0, 2, 4, 6, 8 -> classe 0
1, 3, 5, 7, 9 -> classe 1
```

Com a divisão 80/20, são 1.437 amostras de treino, distribuídas em aproximadamente 287 amostras por cliente.

### Modelo treinado no servidor

Escolhi `RandomForestClassifier` sem nenhum motivo em particular, já que os modelos treinam rapidamente para esse tamanho de dataset. O servidor treina um modelo por cliente para a detecção e depois treina os modelos finais diretamente sobre os dados recebidos, conforme exigido pelo enunciado.

### Particionamento dos dados

O servidor faz uma única divisão estratificada e distribui partições disjuntas. Assim, todos os cenários usam exatamente os mesmos dados por cliente; somente o comportamento dos clientes é alterado, tornando a comparação mais justa.

### Critério de identificação

O critério escolhido foi avaliar o modelo treinado com os dados de cada cliente no conjunto de teste reservado e aceitar somente modelos com desempenho maior que:

```text
1 / número de classes = 1 / 2 = 0.5
```

Essa é uma heurística simples e deliberadamente ingênua, adequada ao objetivo de observar como diferentes ataques afetam a agregação. Ela não garante a identificação de todo cliente malicioso: por exemplo, corromper uma única feature pode preservar informação suficiente para o modelo continuar acima de 0.5.

### Ataques bizantinos

Inicialmente foi usado ruído aditivo, mas ele preservava parte importante do sinal das imagens. A implementação atual substitui features por valores aleatórios uniformes entre `0` e `16`, faixa dos pixels do `load_digits`:

- tipo 1 substitui todas as features;
- tipo 2 substitui as últimas 8 features, correspondentes a uma linha da imagem 8x8. Isso foi uma tentativa, embora não sucedida, de aumentar o impacto desse tipo de falha, mas afetar uma linha inteira mesmo assim não surtiu muito efeito na dificuldade da tarefa;
- tipo 3 inverte os rótulos `0` e `1`.

As sementes aleatórias são derivadas do `client_id`, permitindo reproduzir os experimentos.

### Comparação com e sem detecção

Para cada cenário, o servidor calcula duas acurácias no mesmo conjunto de teste:

- **com detecção:** treinamento no servidor usando somente os dados dos clientes aceitos;
- **sem detecção:** treinamento no servidor usando os dados dos cinco clientes.

Essa comparação mostra o impacto do mecanismo de identificação na qualidade do modelo global.

## Arquivos principais

- `ml.proto`: contrato gRPC;
- `mlservidor.py`: distribuição dos dados, detecção e treinamento;
- `mlcliente.py`: aplicação dos comportamentos e envio dos dados;
- `experimento.py`: execução automatizada com Ray;
- `output/`: relatórios CSV e Markdown.
