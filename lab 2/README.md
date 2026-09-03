# Laboratório 2 - Clientes Bizantinos no Aprendizado Federado

Este projeto simula cinco clientes federados que treinam modelos localmente e enviam seus parâmetros ao servidor por gRPC. O servidor avalia os modelos recebidos, identifica updates que não superam a acurácia de um chute aleatório e agrega os modelos aprovados.

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
| 2 | Substitui uma feature aleatória por valores aleatórios |
| 3 | Inverte os rótulos binários |

Exemplo:

```powershell
python mlcliente.py 4 3
```

## Organização do sistema

1. O servidor carrega o dataset e separa 80% para treino e 20% para teste.
2. Os dados de treino são distribuídos em cinco partições disjuntas, identificadas pelo `client_id`.
3. Cada cliente recebe somente sua partição por meio do RPC `GetTrainingData`.
4. Cada cliente treina localmente uma `LogisticRegression`.
5. O cliente envia `coef_` e `intercept_` no RPC `SubmitModelUpdate`.
6. O servidor espera os cinco updates antes de processá-los.
7. Cada modelo é avaliado no conjunto de teste reservado.
8. O servidor aceita modelos com acurácia estritamente maior que a de um chute aleatório.
9. Os pesos aceitos são agregados pela média.

## Decisões tomadas

### Comunicação via gRPC

O Lab 1 já fornecia a comunicação gRPC e a estrutura protobuf. Mantivemos essa base e acrescentamos mensagens para solicitar dados de treino e enviar parâmetros dos modelos, evitando que os clientes transmitam novamente seus dados depois do treinamento local.

### Cinco processos de cliente com Ray

O enunciado pede múltiplos clientes e apresenta Ray como ferramenta de paralelização. Por isso, o experimento cria cinco `ClienteActor` com `@ray.remote`. Cada actor representa um cliente independente e executa seu treinamento em um worker separado.

### Dataset `load_digits`

O Iris era pequeno para observar o efeito dos ataques. O `load_digits` possui 1.797 amostras e 64 features. Os rótulos foram convertidos em um problema binário:

```text
0, 2, 4, 6, 8 -> classe 0
1, 3, 5, 7, 9 -> classe 1
```

Com a divisão 80/20, são 1.437 amostras de treino, distribuídas em aproximadamente 287 amostras por cliente.

### Modelo com pesos agregáveis

A árvore de decisão do Lab 1 não possui pesos simples que possam ser agregados de forma direta. Ela foi substituída por `LogisticRegression`, que fornece `coef_` e `intercept_`, permitindo enviar e calcular a média dos parâmetros.

### Particionamento dos dados

O servidor faz uma única divisão estratificada e distribui partições disjuntas. Assim, todos os cenários usam exatamente os mesmos dados por cliente; somente o comportamento dos clientes é alterado, tornando a comparação mais justa.

### Critério de identificação

O critério escolhido foi avaliar o modelo de cada cliente no conjunto de teste reservado e aceitar somente modelos com desempenho maior que:

```text
1 / número de classes = 1 / 2 = 0.5
```

Essa é uma heurística simples e deliberadamente ingênua, adequada ao objetivo de observar como diferentes ataques afetam a agregação. Ela não garante a identificação de todo cliente malicioso: por exemplo, corromper uma única feature pode preservar informação suficiente para o modelo continuar acima de 0.5.

### Ataques bizantinos

Inicialmente foi usado ruído aditivo, mas ele preservava parte importante do sinal das imagens. A implementação atual substitui features por valores aleatórios uniformes entre `0` e `16`, faixa dos pixels do `load_digits`:

- tipo 1 substitui todas as features;
- tipo 2 substitui uma única feature escolhida aleatoriamente;
- tipo 3 inverte os rótulos `0` e `1`.

As sementes aleatórias são derivadas do `client_id`, permitindo reproduzir os experimentos.

### Comparação com e sem detecção

Para cada cenário, o servidor calcula duas acurácias no mesmo conjunto de teste:

- **com detecção:** média dos pesos dos modelos aceitos;
- **sem detecção:** média dos pesos de todos os cinco modelos.

Essa comparação mostra o impacto do mecanismo de identificação na qualidade do modelo global.

## Arquivos principais

- `ml.proto`: contrato gRPC;
- `mlservidor.py`: distribuição dos dados, avaliação e agregação;
- `mlcliente.py`: treinamento local e aplicação dos comportamentos;
- `experimento.py`: execução automatizada com Ray;
- `output/`: relatórios CSV e Markdown.
