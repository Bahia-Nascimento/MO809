# Laboratório 5 — Split Learning U-Shaped

TensorFlow/Keras treina os três blocos na GPU do WSL. O gRPC transporta ativações e
gradientes; Ray mantém cada cliente em um processo independente. Cada cliente
possui seus próprios M1/M3 e dados locais, enquanto todos colaboram com um único M2.

## Executar

No terminal WSL:

```bash
cd "/home/bahia/MO809/lab 5"
source ../venv-gpu/bin/activate
# Nova execução completa: dois clientes, dez épocas, batch 64.
python experimento.py --output output/nova_execucao
```

O ambiente existente já contém as dependências. Para recriá-lo, use Python 3.9 e
`python -m pip install -r requirements.txt`. O primeiro uso baixa o CIFAR-10 pelo
Keras. Não use o ambiente Windows nem o `requirements.txt` da raiz.

O comando inicia e encerra servidor e clientes, prepara os dados, treina, seleciona
o checkpoint pela validação, avalia o teste e gera os gráficos. Um diretório já
usado para treinamento é recusado para preservar os resultados.

```bash
# Execução curta com três clientes e um último lote incompleto.
python experimento.py --output output/smoke --clientes 3 --epocas 1 \
  --limite-treino 330 --limite-teste 90 --batch-size 64

# Testes locais independentes do download.
python -m unittest -v test_lab5.py

# Teste de integração na GPU com três clientes e dados sintéticos pequenos.
python test_integracao_gpu.py

# Regenerar os stubs se modificar o protocolo.
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. splitlearning.proto

# Recriar o relatório de uma execução concluída.
python relatorio.py output/nova_execucao

# Conferir dados e reproduzir o teste em novos processos Ray/gRPC.
python verificar_checkpoint.py --output output/nova_execucao
```

## Modelos e treinamento

| Bloco | Local | Camadas | Saída por imagem |
|---|---|---|---|
| M1 | Cada cliente | Conv2D(32,3), MaxPool(2), Conv2D(64,3), Flatten, Dense(128) | 128 |
| M2 | Servidor compartilhado | Dense(64, ReLU) | 64 |
| M3 | Cada cliente | Dense(10, softmax) | 10 probabilidades |

As camadas internas de M1 usam ReLU. M1/M3 começam com a mesma inicialização entre
clientes, mas seus pesos e otimizadores evoluem independentemente; não há agregação
de pesos locais. M2 recebe atualizações de todos os clientes. Adam usa taxa 0,001
nos três blocos; a arquitetura e a taxa seguem a configuração didática do laboratório.
`--learning-rate` permite outro experimento, sem alterar o conjunto de teste.

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

Cada processo lê `dados/cliente_N.npz`. O preparo central desses arquivos simula a
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
ajustá-lo. A configuração validada está em `output/processos.json`. CUDA também
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
| `relatorio.py` | Curvas e resumo dos resultados |
| `test_lab5.py` | Equivalência matemática, RPCs e invariantes dos dados |
| `test_integracao_gpu.py` | Três clientes reais na GPU, lotes incompletos e checkpoint |
| `verificar_checkpoint.py` | Auditoria dos dados e nova inferência Ray/gRPC dos modelos salvos |

Resultados medidos e gráficos ficam em [output/README.md](output/README.md).
`historico.csv` reúne épocas; `cliente_N/results.csv` contém cada lote, loss,
acurácia, tempos e bytes Protobuf. `config.json` registra parâmetros, versões e
status; `processos.json` registra PIDs e dispositivos; `particoes.json` registra
tamanhos, distribuição de classes e hashes dos dados locais.

As observações sobre o HTML estão reunidas em
[ANALISE_DO_HTML.md](ANALISE_DO_HTML.md).
