"""Relatório reproduzível a partir de métricas efetivamente registradas."""
import argparse
import json
import os
from pathlib import Path


def gerar(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out = Path(output)
    destino = Path(os.path.relpath(out.resolve(), Path(__file__).resolve().parent)).as_posix()
    resultados = json.loads((out / "resultados.json").read_text())
    config = json.loads((out / "config.json").read_text())
    dados = json.loads((out / "particoes.json").read_text())
    h = resultados["epocas"]
    epocas = [r["epoch"] for r in h]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for prefix, label in (("train", "Treino"), ("val", "Validação")):
        ax[0, 0].plot(epocas, [r[f"{prefix}_loss"] for r in h], marker="o", label=label)
        ax[0, 1].plot(epocas, [100 * r[f"{prefix}_accuracy"] for r in h], marker="o", label=label)
    ax[0, 0].set_ylabel("Entropia cruzada média")
    ax[0, 1].set_ylabel("Acurácia (%)")
    ax[1, 0].plot(epocas, [r["seconds"] for r in h], label="Época inteira")
    ax[1, 0].plot(epocas, [r["rpc_seconds"] for r in h], label="Chamadas gRPC de treino")
    ax[1, 0].set_ylabel("Tempo (s)")
    for key, label in (("tx_bytes", "Cliente → servidor"), ("rx_bytes", "Servidor → cliente")):
        ax[1, 1].plot(epocas, np.cumsum([r[key] for r in h]) / 2**20, label=label)
    ax[1, 1].set_ylabel("Protobuf de treino acumulado (MiB)")
    for axis in ax.flat:
        axis.set_xlabel("Época")
        axis.set_xticks(epocas)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.savefig(out / "treinamento.png", dpi=160)
    plt.close(fig)
    smoke = sum(c["treino"] for c in dados["clientes"]) < 45000 or resultados["teste"]["n"] < 10000
    linhas = ["# Resultados — Laboratório 5", "",
        "Execução reduzida para validar o pipeline; não representa avaliação final." if smoke else
        "Execução com os conjuntos completos de treino, validação e teste do CIFAR-10.", "",
        f"- Clientes: {config['clientes']}; épocas executadas: {len(h)} "
        f"(limite {config['epocas']}); batch: {config['batch_size']}.",
        f"- Treino: {sum(c['treino'] for c in dados['clientes'])}; validação: "
        f"{sum(c['validacao'] for c in dados['clientes'])}; teste: {resultados['teste']['n']} imagens.",
        f"- Melhor época pela loss de validação: **{resultados['melhor_epoca']}**.",
        f"- Teste: **{100 * resultados['teste']['accuracy']:.2f}%** de acurácia; "
        f"loss **{resultados['teste']['loss']:.4f}**.",
        f"- Atualizações de M2: {resultados['updates_m2']}; "
        f"tempo incluindo inicialização dos processos: {resultados['segundos'] / 60:.2f} min.", "",
        "| Cliente | Exemplos de teste | Acurácia | Loss |", "|---|---:|---:|---:|"]
    for c in resultados["teste_clientes"]:
        linhas.append(f"| {c['client_id']} | {c['n']} | {100*c['accuracy']:.2f}% | {c['loss']:.4f} |")
    linhas += ["", "![Curvas de treinamento](treinamento.png)", "",
        "Os três blocos executaram na GPU. "
        "Cada cliente tem M1/M3 próprios e usa sua partição de teste. A acurácia agregada "
        "é ponderada pelo número de exemplos, não é a avaliação de uma única rede global.", "",
        "A loss de treino reúne lotes durante atualizações; a validação usa os pesos fixos "
        "ao final da época. As curvas não medem exatamente o mesmo estado do modelo. "
        "O teste foi avaliado após restaurar o checkpoint conjunto da melhor época, "
        "sem orientar a seleção. Pequenas variações numéricas podem ocorrer ao mudar "
        "o hardware ou o modo de execução.", "",
        "`rpc_seconds` mede o tempo observado das duas chamadas, incluindo serviço e "
        "serialização; não isola a latência de rede. Os bytes medem mensagens Protobuf "
        "e excluem cabeçalhos de transporte. Execução em loopback não caracteriza uma WAN.", "",
        f"Artefatos (caminhos relativos à pasta `lab 5`): `{destino}/config.json`, "
        f"`{destino}/particoes.json`, `{destino}/historico.csv`, `{destino}/resultados.json`, "
        f"`{destino}/M2.keras`, `{destino}/cliente_N/M1.keras`, `{destino}/cliente_N/M3.keras` "
        f"e `{destino}/cliente_N/results.csv`. Os modelos formam um checkpoint de inferência; "
        "o estado completo dos otimizadores para retomada não é salvo.", ""]
    if config.get("patience", 0):
        linhas += [f"Early stopping coordenado: paciência de {config['patience']} épocas, "
            "monitorando a menor loss de validação agregada e restaurando o conjunto completo. "
            f"Dropout: {config.get('dropout', 0)}; pooling adicional: {config.get('pooling_extra', False)}; "
            f"Batch Normalization: {config.get('batch_normalization', False)}; "
            f"taxa de aprendizado: {config['learning_rate']}.", ""]
    melhor = next(r for r in h if r["epoch"] == resultados["melhor_epoca"])
    if h[-1]["val_loss"] > melhor["val_loss"] and h[-1]["train_loss"] < melhor["train_loss"]:
        linhas += ["## Interpretação", "",
            f"Há sinais de sobreajuste após a época selecionada: a loss de treino "
            f"caiu de {melhor['train_loss']:.4f} para {h[-1]['train_loss']:.4f}, enquanto "
            f"a de validação passou de {melhor['val_loss']:.4f} para {h[-1]['val_loss']:.4f}. "
            "Por isso, a entrega usa o checkpoint escolhido pela validação, preservando "
            f"a configuração registrada e o limite de {config['epocas']} épocas desta execução.", ""]
    if config.get("dropout", 0) or config.get("batch_normalization", False):
        camadas = [nome for chave, nome in (("dropout", "Dropout"),
                   ("batch_normalization", "Batch Normalization")) if config.get(chave)]
        linhas += [f"Camadas com comportamento distinto entre treino e inferência nesta execução: "
            f"{', '.join(camadas)}. "
            "A loss de treino registrada usa regularização ativa e pesos em atualização; "
            "por isso, a diferença direta entre treino e validação não isola sobreajuste.", ""]
    if (out / "busca.json").exists():
        busca = json.loads((out / "busca.json").read_text())
        linhas += ["## Tentativa de reduzir o overfitting", "",
            f"Realizamos uma busca com Keras Tuner RandomSearch de {len(busca['trials'])} "
            "configurações, variando Dropout, pooling adicional, Batch Normalization e taxa "
            "de aprendizado. As tentativas usaram as mesmas partições e semente, com "
            f"limite de {busca['epocas_maximas']} épocas e paciência {busca['patience']}. "
            "A seleção usou somente a loss de validação; o teste foi reservado para "
            "o treino final independente da configuração selecionada.", "",
            "A tentativa não foi suficiente para eliminar os sinais de sobreajuste. "
            "O checkpoint entregue corresponde à melhor época pela validação, e não "
            "à última época treinada. As configurações e métricas resumidas da busca "
            "estão em [busca.json](busca.json). Uma execução por configuração não mede "
            "a variabilidade entre sementes nem isola o efeito de cada técnica.", ""]
    (out / "README.md").write_text("\n".join(linhas), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    gerar(parser.parse_args().output)
