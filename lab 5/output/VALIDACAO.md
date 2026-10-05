# Validação da entrega — 24/09/2026

- **7 testes matemáticos/protocolo passaram.** Dois passos de SGD via gRPC
  coincidiram com os da rede completa em CPU: erro máximo observado nos gradientes
  dos dois cortes igual a zero; pesos finais dos três blocos também conferidos.
  Há testes de partições, mensagens inválidas, concorrência, reenvio, timeout,
  avaliação sem treinamento e serialização.
- **1 teste de integração GPU passou.** Três clientes Ray e um servidor, todos em
  processos distintos na GPU, completaram seis lotes sintéticos, incluindo lotes
  incompletos, salvamento/restauração e avaliação. Isso verifica a infraestrutura;
  dados sintéticos não medem a qualidade do classificador.
- **Treinamento completo passou.** Dois clientes, dez épocas, 45.000 imagens de
  treino por época e 7.040 atualizações de M2, exatamente a quantidade esperada.
- **Auditoria dos arquivos passou.** Hashes dos dados conferidos, índices sem
  sobreposição entre partições e checkpoint recarregado em novos processos
  Ray/gRPC. Os 3.199 e 3.209 acertos dos clientes foram reproduzidos; diferenças
  absolutas de loss: aproximadamente 1,14 × 10⁻⁷ e zero. Nenhum update foi realizado
  durante essa auditoria.
- `pip check`: nenhuma dependência quebrada no ambiente utilizado.

Comandos, a partir da pasta do laboratório no WSL:

```bash
source ../venv-gpu/bin/activate
python -m unittest -v test_lab5.py
python test_integracao_gpu.py
python verificar_checkpoint.py --output output
```

Evidências: [testes.txt](testes.txt), [processos.json](processos.json),
[config.json](config.json) e [verificacao_checkpoint.json](verificacao_checkpoint.json).

## Reprodução numérica

Uma auditoria adicional executou os três blocos consecutivamente em um único
processo GPU. Ela obteve 6.407 acertos, contra 6.408 do experimento distribuído;
as diferenças de loss por cliente ficaram em aproximadamente 4,1 × 10⁻⁵ e
1,1 × 10⁻⁴. Recriar a arquitetura e carregar os mesmos pesos produziu o mesmo
resultado dessa execução local; copiar as ativações via NumPy nos cortes também.

Ao repetir o **caminho distribuído original**, os acertos foram reproduzidos e a
loss coincidiu dentro da tolerância numérica do teste. Isso confirma a consistência
do checkpoint com o experimento entregue. Não foi isolada a operação GPU responsável
pela diferença entre os caminhos; não se afirma reprodução bit a bit ao mudar o
modo de execução. A avaliação oficial permanece a distribuída: **64,08%**.

O verificador entregue utiliza novos processos e a mesma comunicação gRPC do
treinamento. A igualdade matemática dos gradientes é testada separadamente em CPU,
onde a comparação controlada não apresentou diferença observável.
