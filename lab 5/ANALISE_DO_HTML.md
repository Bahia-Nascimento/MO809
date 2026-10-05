# Análise do HTML — Laboratório 5

Leitura crítica do arquivo `Laboratório 5.html` fornecido para o desenvolvimento.
Este documento concentra as divergências, abstrações e decisões complementares;
os comentários no código explicam o funcionamento da implementação.

## 1. O tutorial e o entregável usam fluxos diferentes

O tutorial mostra Split Learning tradicional: cliente envia ativações **e rótulos**;
servidor calcula a loss. O entregável pede **U-Shaped**, no qual o cliente também
possui o classificador final e calcula a loss. Por isso, não basta executar o cliente
original várias vezes.

Na implementação, M1/M3 são locais e M2 é compartilhado. O protocolo remove rótulos,
loss e acurácia e possui dois RPCs de treinamento: `Forward` devolve ativações de M2;
`Backward` recebe dL/db e devolve dL/da. O cliente calcula e registra as métricas.

## 2. Ativações não são pesos

A introdução afirma que são transmitidos os “pesos de última camada”. Os exemplos
e o algoritmo de Split Learning trocam **ativações da camada de corte** no forward
e **gradientes dessas ativações** no backward. Os pesos ficam no processo que
mantém cada bloco. Esta implementação não faz FedAvg nem troca pesos entre clientes.

Também não tratamos a ausência de dados brutos como garantia de privacidade:
ativações e gradientes podem revelar informação. O próprio HTML menciona ataques
de inversão. U-Shaped conserva os rótulos localmente, sem provar que eles sejam
impossíveis de inferir a partir das mensagens.

## 3. A dimensão de saída de M1 está inconsistente

Com as opções padrão das convoluções, a sequência apresentada tem as dimensões:

```text
(B,32,32,3) -> Conv32 -> (B,30,30,32)
             -> Pool -> (B,15,15,32)
             -> Conv64 -> (B,13,13,64)
             -> Dense128 -> (B,13,13,128)
```

`Dense` em um tensor 4D atua sobre o último eixo; não transforma automaticamente
as dimensões espaciais em um único vetor. Achatar a última saída geraria 21.632
valores por imagem, incompatíveis com a entrada `(128,)` declarada no servidor.

Inserimos `Flatten` **antes** da Dense(128). M1 passa a produzir exatamente 128
componentes; M2 produz 64; M3 produz as 10 probabilidades de classe. Conservamos os
filtros, kernels, pooling e unidades apresentados. Flatten aumenta o número de
pesos em relação a uma alternativa com GlobalAveragePooling; essa alternativa
alteraria a arquitetura didática e não foi usada.

## 4. A criação do modelo do servidor é abstraída

O HTML define `ServerModel`, mas instancia `create_server_model((128,))`, função
que não aparece no trecho. Implementamos as três funções de criação explicitamente
pela Functional API. A Dense(64, ReLU) pertence a M2 e a Dense(10, softmax) a M3.
As funções retornam modelos construídos, com dimensões verificáveis e serialização
em `.keras` sem objetos personalizados.

## 5. Gradientes exigem o forward correspondente

O trecho tradicional resolve o backward do servidor dentro de um único RPC. No
formato U-Shaped, M2 precisa conservar seu `GradientTape` entre forward e backward.
Converter tensores para NumPy/Protobuf interrompe o grafo automático; a conexão
matemática entre blocos é reconstruída com `output_gradients`.

Calculamos os gradientes de entrada e de pesos em uma única chamada por tape.
`persistent=True` não é necessário. A entrada de M2 e a entrada de M3 recebem
`tape.watch`, pois são tensores recebidos, não variáveis treináveis.

O HTML descreve atualizar M3 antes de enviar seu gradiente. Nosso cliente calcula
esses gradientes no mesmo ponto, mas aplica os updates de M1/M3 após a resposta de
M2. Isso preserva o resultado de um passo síncrono: todas as derivadas correspondem
aos mesmos pesos usados no forward. Não acrescentamos outra divisão pelo batch,
porque a loss já é uma média.

## 6. Múltiplos clientes precisam de uma política para M2

O enunciado pede vários clientes e porções diferentes dos dados, mas não define
compartilhamento de M1/M3, ordem de atendimento ou sincronização do otimizador.
Adotamos M1/M3 independentes por cliente e um M2 compartilhado. Essa decisão produz
uma família de modelos locais que compartilha o bloco intermediário, não uma única
rede global com todos os pesos idênticos.

Ray mantém clientes em processos separados. O orquestrador alterna lotes completos
entre eles, rotacionando quem começa a época. M2 nunca recebe atualização de outro
lote entre o forward e seu backward. Um simples lock em cada RPC não garantiria
isso, porque o lock seria liberado entre as duas chamadas; por isso o servidor
também mantém um lote pendente com cliente, passo e token, rejeitando interferências.

Essa política privilegia correção e clareza. Não promete aceleração por paralelismo.
Não implementamos cópias versionadas de M2, treinamento assíncrono ou agregação
federada, pois exigiriam outro algoritmo.

## 7. Partições e avaliação não estão completas no tutorial

Carregar CIFAR-10 inteiro em cada processo não satisfaria a exigência de porções
diferentes. Preparamos arquivos locais disjuntos com divisão estratificada.
Reservamos 10% do conjunto oficial de treino para validação; o teste oficial é
separado e só participa da avaliação final. Essa proporção é uma decisão explícita
de implementação, não um requisito textual do HTML.

Usamos 10 épocas, batch 64 e Adam(0,001), seguindo a configuração do exemplo.
Selecionamos a época pela menor loss de validação e salvamos todos os blocos
conjuntamente. Não introduzimos busca de hiperparâmetros: o laboratório pede
desenvolver e verificar a comunicação de treinamento, e a arquitetura foi mantida
próxima à apresentada.

O laço `N // batch_size` do HTML descarta o restante. Usamos todos os exemplos,
incluindo o último lote menor, e embaralhamos o treino a cada época. Agregamos loss
e acurácia por quantidade de exemplos para que lotes/clientes de tamanhos diferentes
tenham peso correto.

## 8. Métricas e gráficos requerem contexto

`SparseCategoricalAccuracy` é um acumulador: reutilizá-lo sem reset não fornece a
acurácia isolada do lote. Calculamos acertos por lote e depois agregamos. O contador
global incrementado no servidor do HTML conta chamadas/lotes, não épocas.

`array.nbytes` mede o payload de um array, não toda a mensagem enviada. Registramos
`ByteSize()` dos pedidos e respostas de ambos os RPCs. Ainda assim, esse número
exclui cabeçalhos gRPC/HTTP2/TCP. O tempo de RPC inclui serviço e comunicação; não é
uma medição isolada da latência de rede. A demonstração local usa loopback.

Os gráficos usam as épocas realmente executadas, acurácia em porcentagem quando o
eixo indica `%`, bytes em MiB e sem posições de ticks fixadas para outro experimento.
Treino é medido durante as atualizações; validação usa um modelo fixo ao final de
cada época, portanto a comparação exige esse cuidado.

## 9. Robustez acrescentada para executar o exercício

Validamos dimensões, finitude, sequência e identidade dos lotes, adicionamos
deadlines e evitamos reaplicar updates em mensagens repetidas. Um timeout invalida
a sessão; não há garantia de commit atômico entre processos nem recuperação de
resposta perdida. O programa interrompe a execução nesses casos, sem retry de
treinamento. Os checkpoints entregues servem para inferência, não retomada de Adam.

A memória de GPU é limitada por processo, pois frações de GPU do Ray não impõem
limites físicos de VRAM. Treino de M1, M2 e M3 ocorre na GPU; os testes matemáticos
pequenos usam CPU. O experimento verifica PIDs distintos e dispositivos efetivos.

## Verificação e referências técnicas

O teste principal compara dois passos completos via gRPC com a rede M3(M2(M1(x)))
em um único grafo, partindo dos mesmos pesos. Compara os gradientes dos dois cortes
e os pesos dos três blocos após os updates. Os demais testes cobrem partições,
mensagens inválidas, concorrência, replay, timeout, avaliação e checkpoints.

- [TensorFlow: diferenciação automática](https://www.tensorflow.org/guide/autodiff)
- [TensorFlow: GradientTape e output_gradients](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
- [TensorFlow: configuração de GPU e memória](https://www.tensorflow.org/guide/gpu)
- [Ray: recursos de aceleradores e GPUs fracionárias](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)

As versões efetivamente utilizadas estão no `requirements.txt` e no `config.json`
da execução; páginas de documentação podem descrever versões mais recentes.
