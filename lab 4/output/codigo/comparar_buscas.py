"""Compara uma execução ampliada com a referência usando os mesmos pares.

Usa arquivos já avaliados, sem selecionar modelos a partir do conjunto de teste.
Gera README, JSON e gráficos no diretório da nova execução.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUTPUT, ROOT, salvar_json


def ler(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--referencia", type=Path, default=ROOT / "output_inicial_20260917")
    parser.add_argument("--nova", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ref, nova = args.referencia.resolve(), args.nova.resolve()
    mr, mn = ler(ref / "metadata.json"), ler(nova / "metadata.json")
    hr, hn = ler(ref / "historico.json"), ler(nova / "historico.json")
    trials = ler(nova / "busca.json")
    with np.load(ref / "indices_pares.npz") as a, np.load(nova / "indices_pares.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    sel = mn["selecao"]
    anterior, atual = sel["loss_validacao_referencia"], mn["loss_validacao"]
    ganho = (anterior - atual) / anterior
    resumo = {"loss_validacao_anterior": anterior, "loss_validacao_nova": atual,
              "reducao_relativa": ganho, "mesmos_indices_e_pares": True,
              "criterio_selecao": "menor Contrastive Loss de validacao; teste so apos selecao",
              "referencia": str(ref), "nova": str(nova)}
    salvar_json(nova / "comparacao.json", resumo)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for hist, nome, cor in [(hr, "Inicial", "#777777"), (hn, "Ampliada", "#146b9e")]:
        epocas = np.arange(1, len(hist["val_loss"]) + 1)
        axes[0].plot(epocas, hist["val_loss"], marker=".", label=nome, color=cor)
        melhor = int(np.argmin(hist["val_loss"]))
        axes[0].scatter(melhor+1, hist["val_loss"][melhor], s=90, facecolors="none", edgecolors=cor)
    axes[0].set(title="Validação: checkpoint selecionado em cada execução", xlabel="Época", ylabel="Contrastive Loss")
    axes[0].legend(); axes[0].grid(alpha=.25)
    for bn, cor in [(False, "#146b9e"), (True, "#c36e16")]:
        grupo = [t for t in trials if t["status"] == "COMPLETED" and bool(t["batch_normalization"]) == bn]
        axes[1].scatter([int(t["trial"])+1 for t in grupo], [t["val_loss"] for t in grupo],
                        label="Com BN" if bn else "Sem BN", color=cor)
    axes[1].axhline(anterior, color="gray", linestyle="--", label="Modelo inicial salvo")
    axes[1].set(title="Melhor validação de cada tentativa da nova busca", xlabel="Tentativa", ylabel="Contrastive Loss")
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(nova / "comparacao_validacao.png", dpi=150); plt.close(fig)
    h = mn["hiperparametros"]
    linhas = ["# Busca ampliada — regularização e perda de validação", "",
      f"Perda de validação: **{anterior:.6f} → {atual:.6f}** ({ganho:.2%} de redução relativa).",
      "Comparação realizada nas mesmas imagens, partições e pares, conferidos elemento a elemento.", "",
      "## Protocolo", "",
      f"- RandomSearch: {mn['configuracao']['trials']} configurações, até {mn['configuracao']['epocas_busca']} épocas por tentativa.",
      f"- Retreinamento do vencedor: até {mn['configuracao']['epocas']} épocas; paciência {mn['configuracao']['paciencia']} no EarlyStopping.",
      "- Dropout: 0; 0,1; 0,2; 0,3; 0,4. Batch Normalization opcional antes da ReLU nas duas convoluções e na camada densa.",
      "- Mantidos filtros, tamanhos de camadas e taxas de aprendizado do espaço anterior: 480 combinações possíveis.",
      "- Margem 1, lote 128 e pares fixos: 30.000 treino, 6.000 validação, 10.000 teste.",
      "- O checkpoint da busca e o retreinamento competem por menor val_loss. O teste só é usado após selecionar.",
      f"- Selecionado: **{sel['selecionado']}**, trial {sel['trial_vencedor']}, época {mn['melhor_epoca']}.",
      f"- Melhor checkpoint da busca: {sel['loss_checkpoint_busca']:.6f}; retreinamento: {sel['loss_retreino']:.6f}.",
      f"- Tempo registrado (dados, busca, retreinamento e avaliação): {mn['segundos']/60:.1f} minutos.", "",
      "## Rede escolhida", "", "| Hiperparâmetro | Valor |", "|---|---:|"]
    linhas += [f"| {k} | {v} |" for k,v in h.items()]
    linhas += ["", "## Avaliação", "", "| Métrica | Inicial | Ampliada |", "|---|---:|---:|",
      f"| Loss de validação | {anterior:.6f} | {atual:.6f} |",
      f"| Loss de treino em inferência | {sel['loss_treino_referencia_inferencia']:.6f} | {sel['loss_treino_inferencia']:.6f} |",
      f"| Gap validação − treino em inferência | {anterior-sel['loss_treino_referencia_inferencia']:.6f} | {atual-sel['loss_treino_inferencia']:.6f} |",
      f"| Acurácia de validação | {mr['acuracia_validacao']:.2%} | {mn['acuracia_validacao']:.2%} |",
      f"| Loss de teste | {mr['loss_teste']:.6f} | {mn['loss_teste']:.6f} |",
      f"| Acurácia de teste | {mr['acuracia_teste']:.2%} | {mn['acuracia_teste']:.2%} |",
      f"| Limiar calibrado na validação | {mr['limiar']:.6f} | {mn['limiar']:.6f} |", "",
      "![Comparação de validação](comparacao_validacao.png)", "",
      "![Curva do candidato selecionado](curva_loss.png)", "",
      "A curva de treino é medida durante atualizações, com dropout/BN em modo de treinamento.",
      "A validação usa inferência; portanto, dropout pode fazer a loss de treino parecer maior.",
      "A tabela também avalia o treino em inferência para comparar os conjuntos no mesmo modo.", "",
      "## Tentativas", "", "| Trial | BN | Dropout | Filtros | Densa | Vetor | LR | Melhor val_loss |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for t in sorted(trials, key=lambda t: t["val_loss"] if t["val_loss"] is not None else float("inf")):
        score = f"{t['val_loss']:.6f}" if t['val_loss'] is not None else t['status']
        linhas.append(f"| {t['trial']} | {t['batch_normalization']} | {t['dropout']} | {t['filtros_1']}/{t['filtros_2']} | {t['unidades_densas']} | {t['dimensao_embedding']} | {t['learning_rate']} | {score} |")
    linhas += ["", "## Arquivos e reprodução", "",
      "`siamesa.keras`, `rede_base.keras`, `comparador.keras` e `metadata.json` correspondem ao candidato selecionado.",
      "`vencedor_busca.keras` e `retreino.keras` preservam ambos os candidatos. `historico.json` e",
      "`curva_loss.png` correspondem ao selecionado; `historico_retreino.json` guarda o retreinamento.",
      "`tuner/siamesa/trial_*/historico.csv` contém todas as curvas da busca.",
      "`codigo/` preserva os fontes usados e `codigo_anterior/` guarda os arquivos anteriores às alterações.", "",
      "```bash", 'cd "/home/bahia/MO809/lab 4"', "source ../venv-gpu/bin/activate",
      "python experimento.py",
      "python -m unittest -v test_lab4.py",
      "# Repetir em OUTRO diretório, preservando esta execução:",
      "python treinar.py --output output/repeticao_ampliada --referencia output --trials 24 --epocas-busca 20 --epocas 30 --paciencia 5",
      "```", "", "## Limites da comparação", "",
      "A melhoria mede o conjunto de validação usado na seleção. Não é prova de superioridade para todas as sementes ou dados.",
      "Os hiperparâmetros e o orçamento mudaram juntos: não é uma ablação que isola o efeito causal de BN ou dropout.",
      "Mais tentativas aumentam a adaptação à validação; o teste é apenas uma verificação final e não orientou as escolhas.",
      "Os pares compartilham imagens dentro de cada partição. Não foi estimada incerteza entre execuções.", "",
      "Referências: [Batch Normalization](https://keras.io/api/layers/normalization_layers/batch_normalization/),",
      "[checkpoints do Keras Tuner](https://keras.io/keras_tuner/api/tuners/base_tuner/).", ""]
    gap_ref = anterior - sel["loss_treino_referencia_inferencia"]
    gap_novo = atual - sel["loss_treino_inferencia"]
    linhas += ["## Interpretação e verificação", "",
      f"O gap passou de {gap_ref:.6f} para {gap_novo:.6f}. A redução de val_loss não eliminou o afastamento entre treino e validação.",
      "A comparação da loss de treino usa o checkpoint salvo em modo de inferência; não o último ponto da curva.",
      f"Configuração selecionada: dropout={h['dropout']}, BN={h['batch_normalization']}; o espaço de busca permitia ambos.",
      f"Acurácia de teste mudou {(mn['acuracia_teste']-mr['acuracia_teste'])*100:+.2f} ponto percentual; não foi estimada variabilidade entre sementes.", ""]
    for nome, grupo in [("Com BN", [t for t in trials if t["batch_normalization"]]),
                        ("Com dropout > 0", [t for t in trials if t["dropout"] > 0])]:
        melhor = min(t["val_loss"] for t in grupo if t["val_loss"] is not None)
        linhas.append(f"- {nome}: {len(grupo)} tentativas; melhor val_loss {melhor:.6f} (grupos podem se sobrepor).")
    if (nova / "experimento.json").exists():
        resultados = ler(nova / "experimento.json")
        linhas += ["", f"Integração: {len(resultados)} rodadas gRPC/Ray concluídas; vetores gerados em dois processos distintos.",
                   "Distâncias conferidas numericamente pelo experimento. [Resultados distribuídos](experimento.md).",
                   ]
    if (nova / "testes.log").exists() and "\nOK\n" in (nova / "testes.log").read_text():
        linhas.append("Os testes passaram; consulte testes.log para a contagem e os casos executados.")
    linhas.append("")
    (nova / "README.md").write_text("\n".join(linhas), encoding="utf-8")
    print(json.dumps(resumo, indent=2))


if __name__ == "__main__":
    main()
