"""Busca de hiperparâmetros, treinamento, calibração e avaliação (nessa ordem).

Execute no WSL: python treinar.py. A primeira execução baixa o Fashion MNIST.
O Tuner testa arquiteturas; os pesos de cada uma são aprendidos por backpropagation.
"""
import argparse
import csv
import platform
import shutil
import time
from pathlib import Path

from config import OUTPUT, configurar_tensorflow, hash_modelo, salvar_json


def calibrar_limiar(distancias, labels):
    """Maximiza a acurácia de validação para a regra similar = distância <= limiar.

    Avalia os cortes entre distâncias distintas em O(N log N). Empates na distância
    ficam sempre juntos. Em empate de acurácia, escolhe o menor limiar.
    """
    import numpy as np
    d = np.asarray(distancias, dtype="float64").reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    if len(d) != len(y) or not len(d) or not np.isfinite(d).all():
        raise ValueError("Distâncias/rótulos inválidos para calibração.")
    ordem = np.argsort(d)
    d, y = d[ordem], y[ordem]
    finais = np.r_[np.flatnonzero(np.diff(d) != 0), len(d) - 1]
    # Antes do primeiro corte, todos são classificados como diferentes.
    acertos = np.sum(y == 0) + np.cumsum(np.where(y == 1, 1, -1))
    candidatos = np.r_[np.nextafter(d[0], -np.inf), d[finais]]
    pontos = np.r_[np.sum(y == 0), acertos[finais]]
    melhor = int(np.argmax(pontos))
    return float(candidatos[melhor]), float(pontos[melhor] / len(y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--epocas-busca", type=int, default=20)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--pares-treino", type=int, default=30000)
    parser.add_argument("--pares-validacao", type=int, default=6000)
    parser.add_argument("--pares-teste", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--paciencia", type=int, default=5)
    args = parser.parse_args()
    if min(args.trials, args.epocas_busca, args.epocas, args.batch_size, args.paciencia) < 1:
        parser.error("Trials, épocas e batch size devem ser positivos.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Evita sobrescrever silenciosamente resultados e misturar buscas incompatíveis.
    if (out / "metadata.json").exists() or (out / "tuner").exists():
        parser.error("Diretório já usado: escolha outro --output para nova execução.")
    tf = configurar_tensorflow(gpu=not args.cpu, seed=args.seed)
    import numpy as np
    import keras_tuner as kt
    import matplotlib
    matplotlib.use("Agg")  # Gera PNG no WSL sem precisar abrir uma janela.
    import matplotlib.pyplot as plt
    from dados import CLASSES, criar_pares, dataset_pares, separar_indices
    from modelo import construir_para_tuner
    inicio = time.monotonic()
    print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    (x, y), (xt, yt) = tf.keras.datasets.fashion_mnist.load_data()
    it, iv = separar_indices(y, args.seed)
    splits = [(x[it], y[it]), (x[iv], y[iv]), (xt, yt)]
    quantidades = [args.pares_treino, args.pares_validacao, args.pares_teste]
    pares = [criar_pares(rotulos, n, args.seed + i) for i, ((_, rotulos), n)
             in enumerate(zip(splits, quantidades))]
    datasets = [dataset_pares(imagens, p, args.batch_size, treino=(i == 0), seed=args.seed)
                for i, ((imagens, _), p) in enumerate(zip(splits, pares))]
    treino, validacao, teste = datasets
    # Clientes recebem apenas imagens de teste. Rótulos servem só ao relatório local.
    np.savez_compressed(out / "dados_clientes.npz", imagens=xt, labels=yt)
    class BuscaComHistorico(kt.RandomSearch):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            # Preserva cada curva, inclusive as oscilações das configurações com BN.
            callbacks = list(fit_kwargs.pop("callbacks", []))
            callbacks.append(tf.keras.callbacks.CSVLogger(str(Path(self.get_trial_dir(trial.trial_id)) / "historico.csv")))
            return super().run_trial(trial, *fit_args, callbacks=callbacks, **fit_kwargs)

    tuner = BuscaComHistorico(construir_para_tuner, objective="val_loss",
                           max_trials=args.trials, seed=args.seed,
                           directory=str(out / "tuner"), project_name="siamesa")
    tuner.search(treino, validation_data=validacao, epochs=args.epocas_busca,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.paciencia)], verbose=2)
    hp = tuner.get_best_hyperparameters(1)[0]
    salvar_json(out / "melhores_hiperparametros.json", hp.values)
    trials = [{"trial": t.trial_id, "status": t.status, "val_loss": t.score, **t.hyperparameters.values}
              for t in tuner.oracle.trials.values()]
    salvar_json(out / "busca.json", trials)
    trial_vencedor = tuner.oracle.get_best_trials(1)[0]
    checkpoint = tuner.get_best_models(1)[0]
    checkpoint.save(out / "vencedor_busca.keras")
    loss_checkpoint = float(checkpoint.evaluate(validacao, verbose=0))
    del checkpoint
    # Recomeça o treinamento com os parâmetros vencedores. Teste ainda não é usado.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    modelo = construir_para_tuner(hp)
    history = modelo.fit(treino, validation_data=validacao, epochs=args.epocas,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.paciencia,
                                   restore_best_weights=True)], verbose=2)
    loss_retreino = float(modelo.evaluate(validacao, verbose=0))
    selecionado = "retreino"
    if loss_checkpoint < loss_retreino:
        # Reinicializar pesos pode piorar o resultado; preservamos o melhor candidato
        # por val_loss. O teste ainda não participa desta escolha.
        modelo = tf.keras.models.load_model(out / "vencedor_busca.keras")
        selecionado = "checkpoint_busca"
        with (Path(tuner.get_trial_dir(trial_vencedor.trial_id)) / "historico.csv").open() as f:
            linhas = list(csv.DictReader(f))
        history.history = {k: [float(row[k]) for row in linhas] for k in ["loss", "val_loss"]}
    loss_validacao = float(modelo.evaluate(validacao, verbose=0))
    # Dropout/BN mudam entre treino e inferência. Medimos também treino em modo de
    # inferência para uma comparação justa com a validação (além da curva do fit).
    loss_treino_inferencia = float(modelo.evaluate(treino, verbose=0))
    selecao = {"selecionado": selecionado, "trial_vencedor": trial_vencedor.trial_id,
               "loss_checkpoint_busca": loss_checkpoint, "loss_retreino": loss_retreino,
               "loss_validacao": loss_validacao, "loss_treino_inferencia": loss_treino_inferencia}
    # Distância pequena significa similaridade alta; não é uma probabilidade.
    dval = modelo.predict(validacao, verbose=0).ravel()
    limiar, acc_val = calibrar_limiar(dval, pares[1][2])
    dtest = modelo.predict(teste, verbose=0).ravel()
    verdade = pares[2][2].ravel().astype(bool)
    pred = dtest <= limiar
    matriz = [[int(np.sum((verdade == real) & (pred == previsto))) for previsto in [False, True]]
              for real in [False, True]]
    base = modelo.get_layer("rede_base")
    base.save(out / "rede_base.keras")
    modelo.get_layer("comparador").save(out / "comparador.keras")
    modelo.save(out / "siamesa.keras")
    # Confere a serialização segura e a igualdade entre cálculo local e separado.
    recarregado = tf.keras.models.load_model(out / "siamesa.keras", compile=False)
    xb, _ = next(iter(teste))
    np.testing.assert_allclose(modelo(xb, training=False), recarregado(xb, training=False), atol=1e-6)
    metadata = {"model_id": hash_modelo(out / "rede_base.keras"), "dimensao": int(base.output_shape[-1]),
                "limiar": limiar, "margem": 1.0, "convencao": "1=mesma classe; 0=classes diferentes",
                "hiperparametros": hp.values, "acuracia_validacao": acc_val,
                "selecao": selecao, "loss_validacao": loss_validacao,
                "acuracia_teste": float(np.mean(pred == verdade)), "matriz_confusao_teste": matriz,
                "loss_teste": float(modelo.evaluate(teste, verbose=0)),
                "imagens": {"treino": len(it), "validacao": len(iv), "teste": len(xt)},
                "pares": dict(zip(["treino", "validacao", "teste"], quantidades)),
                "epocas_executadas": len(history.history["loss"]),
                "melhor_epoca": int(np.argmin(history.history["val_loss"]) + 1),
                "configuracao": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "python": platform.python_version(), "tensorflow": tf.__version__,
                "keras": tf.keras.__version__, "keras_tuner": kt.__version__,
                "gpu": [tf.config.experimental.get_device_details(g).get("device_name", g.name)
                        for g in tf.config.list_physical_devices("GPU")],
                "segundos": time.monotonic() - inicio}
    salvar_json(out / "metadata.json", metadata)
    with (out / "historico.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoca", "loss", "val_loss"])
        writer.writerows(zip(range(1, len(history.history["loss"])+1), history.history["loss"], history.history["val_loss"]))
    fig, ax = plt.subplots(figsize=(11, 4))
    for chave, titulo in [("loss", "Treino"), ("val_loss", "Validação")]:
        ax.plot(range(1, len(history.history[chave])+1), history.history[chave], marker="o", label=titulo)
    ax.set(xlabel="Época", ylabel="Contrastive Loss", title=f"Rede selecionada: {selecionado} | dropout={hp.get('dropout')} | BN={hp.get('batch_normalization')}")
    ax.axvline(np.argmin(history.history["val_loss"]) + 1, color="gray", linestyle=":", label="Checkpoint salvo")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "curva_loss.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for col in range(8):
        a, b = int(pares[2][0][col]), int(pares[2][1][col])
        for row, idx in enumerate([a, b]):
            axes[row,col].imshow(xt[idx], cmap="gray")
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            axes[row,col].set_title(CLASSES[int(yt[idx])], fontsize=10)
        texto = "similar" if pred[col] else "diferente"
        axes[1,col].set_xlabel(f"D={dtest[col]:.3f}\n{texto}", color="green" if pred[col] == verdade[col] else "red")
    fig.suptitle(f"Pares de teste | limiar={limiar:.3f} | verde: acerto; vermelho: erro")
    fig.tight_layout(); fig.savefig(out / "pares_teste.png", dpi=150); plt.close(fig)
    with (out / "arquitetura.txt").open("w", encoding="utf-8") as f:
        modelo.summary(print_fn=lambda linha: f.write(linha + "\n"), expand_nested=True)
    # O diretório interno do Tuner e o checkpoint temporário foram úteis apenas
    # durante a busca. Os resultados consolidados permanecem em busca.json,
    # historico.csv, metadata.json e nos modelos finais.
    (out / "vencedor_busca.keras").unlink(missing_ok=True)
    shutil.rmtree(out / "tuner", ignore_errors=True)
    print("RESULTADO:", metadata, flush=True)


if __name__ == "__main__":
    main()
