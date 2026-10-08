# Laboratório 5 — Split Learning U-Shaped

TensorFlow/Keras treina os três blocos na GPU. O gRPC transporta ativações e
gradientes; Ray mantém cada cliente em um processo independente. Cada cliente
possui seus próprios M1/M3 e dados locais, enquanto todos colaboram com um único M2.

## Preparar o ambiente e executar

Requisitos: Python 3.9, Linux x86_64 e uma GPU NVIDIA compatível com CUDA, com o
driver instalado. Todos os comandos abaixo devem ser executados **a partir da
pasta `lab 5`**; os caminhos são relativos a essa pasta.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confira se o TensorFlow reconhece a GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

A lista deve conter uma GPU. O treinamento exige esse dispositivo. Para treinar
com a configuração entregue (dois clientes, até 20 épocas e lotes de 64 imagens):

```bash
python experimento.py --output output/nova_execucao --epocas 20 --patience 5 --dropout 0.2 --pooling-extra --learning-rate 0.001
```

O comando inicia e encerra servidor e clientes, prepara os dados, treina, seleciona
o checkpoint pela validação, avalia o teste e gera os gráficos. O primeiro uso
baixa o CIFAR-10 pelo Keras; as partições locais são geradas automaticamente.
Um diretório já usado para treinamento é recusado. `output/` contém os resultados
entregues; use `output/nova_execucao` ou outro diretório novo para refazer o experimento.

```bash
# Recriar o relatório de uma execução concluída.
python relatorio.py output/nova_execucao

# Recriar apenas o relatório dos resultados entregues, sem treinar novamente.
python relatorio.py output
```

## Busca com Keras Tuner

Para buscar 12 configurações, usando até 20 épocas e parada após cinco épocas
sem redução da loss de validação agregada:

```bash
python busca.py --trials 12 --epocas 20 --patience 5 --output output/nova_busca
```

O RandomSearch explora Dropout após a Dense de M1 (`0`, `0.1`, `0.2`, `0.3`, `0.4`),
pooling adicional após a segunda convolução, Batch Normalization antes de cada
ReLU convolucional e taxa de aprendizado (`0.001`, `0.0003`). As opções se baseiam
nas técnicas empregadas/exploradas no laboratório 4 e na
[página de redes neurais da disciplina](https://ic.unicamp.br/~allanms/mo809-S22026/labs/redes_neurais/).
O pooling reduz os parâmetros da Dense de M1 sem alterar o vetor de 128 componentes
enviado ao servidor. As estatísticas de Batch Normalization são locais a cada cliente.

Cada tentativa inicia processos novos, com modelos e otimizadores novos, mantendo
as mesmas partições e semente. O treinamento continua sendo distribuído em
TensorFlow/GPU, Ray e gRPC. O controlador do Tuner roda na CPU e recebe a menor
loss de validação do experimento. Os arquivos das tentativas contêm somente treino
e validação. Após a seleção, o vencedor é treinado novamente do zero; somente esse
treino final avalia o conjunto de teste. A parada e o checkpoint são coordenados
para todos os blocos, incluindo as estatísticas móveis de Batch Normalization.

`output/nova_busca/busca.json` reúne configurações e métricas resumidas de todas as
tentativas. Modelos descartados e arquivos temporários do Tuner são removidos;
apenas o conjunto de modelos do treino final é entregue. A busca não oferece
retomada: uma execução interrompida preserva seu resumo, e uma nova busca exige
outro diretório. Para recriar seu relatório final:

```bash
python relatorio.py output/nova_busca
```

Os arquivos `splitlearning_pb2.py` e `splitlearning_pb2_grpc.py` já acompanham a
entrega. Se alterar `splitlearning.proto`, regenere-os com:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. splitlearning.proto
```

Em outro terminal, ative o ambiente novamente com `source .venv/bin/activate`.
Ao terminar, use `deactivate`.

## Modelos e treinamento

| Bloco | Local | Camadas | Saída por imagem |
|---|---|---|---|
| M1 | Cada cliente | Conv2D(32,3), MaxPool(2), Conv2D(64,3), MaxPool(2), Flatten, Dense(128), Dropout(0,2) | 128 |
| M2 | Servidor compartilhado | Dense(64, ReLU) | 64 |
| M3 | Cada cliente | Dense(10, softmax) | 10 probabilidades |

As camadas internas de M1 usam ReLU. M1/M3 começam com a mesma inicialização entre
clientes, mas seus pesos e otimizadores evoluem independentemente; não há agregação
de pesos locais. M2 recebe atualizações de todos os clientes. Adam usa taxa 0,001
nos três blocos. A tabela descreve a configuração final selecionada, sem Batch
Normalization. O comando acima ativa explicitamente o pooling adicional e o
Dropout; sem essas opções, o programa usa a arquitetura inicial do exercício.

```mermaid
sequenceDiagram
    participant C as Cliente i: imagem, rótulo, M1 e M3
    participant S as Servidor: M2 compartilhado
    C->>C: a = M1(imagem), guardar tape1
    C->>S: Forward(a, cliente, passo)
    S->>S: b = M2(a), guardar tape2
    S-->>C: b, token do lote
    C->>C: M3(b), loss e gradientes de M3 e b
    C->>S: Backward(dL/db, token)
    S->>S: Gradientes de M2 e a; atualizar M2
    S-->>C: dL/da
    C->>C: Gradientes de M1; atualizar M1 e M3
```

`GradientTape` registra as operações do forward para derivá-las depois. A conversão
para Protobuf interrompe o grafo automático entre processos. `output_gradients`
introduz o gradiente vindo do outro bloco e aplica a regra da cadeia. A loss é
calculada no cliente, usando a média da entropia cruzada do lote.

O Ray alterna os clientes por lote e muda quem começa a cada época. Cada lote
completa forward e backward antes do próximo: M2 conserva os mesmos pesos durante
o cálculo de seus gradientes. Existem vários processos vivos, com execução de
treinamento sequencial para manter esse contrato; não é treinamento assíncrono.

## Dados e avaliação

Dos 50.000 exemplos de treino oficiais, 45.000 ficam no treino e 5.000 na validação.
Os 10.000 exemplos do teste oficial ficam reservados para a avaliação final.
As três partes são distribuídas de modo estratificado e sem sobreposição entre
clientes. Índices são salvos nos arquivos locais `.npz`; índices do teste referem-se
ao conjunto oficial de teste, não ao array de treino. Semente padrão: 42.

Cada processo lê sua partição em `output/nova_execucao/dados/cliente_N.npz` no
exemplo de execução acima. O preparo central desses arquivos simula a
distribuição de dados nesta única máquina; não é uma fronteira de segurança entre
organizações. O servidor gRPC não abre os arquivos de dados e seu protocolo não
contém imagens ou rótulos. Ativações e gradientes, contudo, não oferecem garantia
formal de privacidade. A demonstração usa gRPC sem TLS restrito ao loopback.

Imagens ficam em `uint8` na RAM e são convertidas para `float32 / 255` por lote.
Todos os exemplos selecionados são usados, inclusive o último lote incompleto.
Cada época embaralha apenas os dados de treino. As métricas são ponderadas pelo
número de exemplos. A seleção usa a menor loss de validação agregada e salva o
conjunto M2 + todos os M1/M3 na mesma época. Somente depois dessa seleção, cada
cliente avalia sua própria partição de teste.

## GPU, comunicação e limites

Cada processo TensorFlow tem teto padrão de 1.024 MiB; `--memoria-mb` permite
ajustá-lo. Os parâmetros da execução entregue estão em `output/config.json`. CUDA também
consome memória fora desse orçamento. Ao aumentar o número de clientes, considere
a VRAM disponível. Ray reserva frações lógicas da GPU para os clientes; o processo
servidor é gerenciado separadamente. O programa verifica o dispositivo efetivo do
forward em todos os processos e falha se não houver GPU.

O servidor escolhe uma porta livre; não encerra servidores de outros laboratórios.
Aceita um lote pendente, valida forma, valores finitos, cliente, passo e token.
Mensagens repetidas não causam uma segunda atualização. A ausência de backward por
60 segundos invalida a sessão quando a próxima chamada chega. Os RPCs também têm
deadline de 60 segundos. Uma falha interrompe o experimento: não há retry automático
nem transação distribuída para recuperar uma resposta perdida após um update.

Checkpoints `.keras` permitem inferência e comparação dos resultados. Não contêm
o estado completo necessário para retomar o treinamento distribuído. Depois de
restaurar os pesos para teste, o servidor e os clientes recusam novo treinamento.

## Arquivos e resultados

| Arquivo | Responsabilidade |
|---|---|
| `dados.py` | Download, seleção e partições locais |
| `modelo.py` | M1, M2 e M3 pela Functional API |
| `cliente.py` | Forward de M1/M3, loss, backward e métricas |
| `servidor.py` | Serviço gRPC, tape remoto, atualização de M2 |
| `splitlearning.proto` | Mensagens e RPCs; stubs gerados pelo protoc |
| `experimento.py` | Processos Ray, ordem dos lotes, seleção e teste |
| `busca.py` | Keras Tuner sobre os experimentos distribuídos e treino final |
| `relatorio.py` | Curvas e resumo dos resultados |
| `config.py` | Configuração de TensorFlow, GPU e comunicação |
| `requirements.txt` | Dependências e versões para criar o ambiente |

**[output/README.md](output/README.md) faz parte da entrega:** apresenta os
resultados finais, os gráficos e a interpretação do treinamento. A execução
entregue atingiu **65,61% de acurácia no teste**, com o checkpoint da época 8;
o treino foi encerrado na época 13 por parada antecipada.

Tentamos reduzir o overfitting com uma busca de 12 configurações envolvendo
Dropout, pooling adicional, Batch Normalization e taxa de aprendizado.
A configuração selecionada usa Dropout 0,2, pooling adicional, sem Batch
Normalization e taxa 0,001. A tentativa não foi suficiente para eliminar os
sinais de sobreajuste, conforme discutido no relatório de resultados.

Os artefatos finais estão organizados em `output/`:

| Arquivo | Conteúdo |
|---|---|
| `output/README.md` | Relatório e discussão dos resultados |
| `output/treinamento.png` | Curvas de loss, acurácia, tempo e comunicação |
| `output/config.json` | Parâmetros, versões e status da execução |
| `output/particoes.json` | Tamanhos, distribuição de classes e hashes das partições |
| `output/historico.csv` | Métricas por época |
| `output/resultados.json` | Resultados agregados e por cliente |
| `output/busca.json` | Configurações, métricas de validação e escolha da busca |
| `output/M2.keras` | Modelo compartilhado selecionado |
| `output/cliente_N/M1.keras` e `output/cliente_N/M3.keras` | Modelos locais selecionados |
| `output/cliente_N/results.csv` | Métricas, tempos e bytes Protobuf por lote |

As cópias locais do CIFAR-10 são recriadas pelo programa e não integram a entrega.

