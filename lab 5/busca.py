"""Keras Tuner sobre o treinamento U-Shaped real, com seleção pela validação."""
import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from config import ROOT, salvar_json

# O controlador do Tuner não treina modelos nem reserva VRAM. Seus subprocessos
# recebem o ambiente original, mantendo a GPU disponível a servidor e clientes.
WORKER_ENV = os.environ.copy()
WORKER_ENV["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras_tuner as kt


def comando_treino(args, output, hp, somente_validacao=False):
    comando = [sys.executable, "-u", str(ROOT / "experimento.py"),
        "--output", str(output), "--epocas", str(args.epocas),
        "--patience", str(args.patience), "--seed", str(args.seed),
        "--clientes", "2", "--batch-size", "64", "--memoria-mb", str(args.memoria_mb),
        "--dropout", str(hp["dropout"]), "--learning-rate", str(hp["learning_rate"])]
    for nome in ("pooling_extra", "batch_normalization"):
        if hp[nome]:
            comando.append("--" + nome.replace("_", "-"))
    if somente_validacao:
        comando.append("--somente-validacao")
    return comando


class BuscaDistribuida(kt.RandomSearch):
    def run_trial(self, trial, args, workspace, output, resumo):
        hp = dict(trial.hyperparameters.values)
        inicio = time.perf_counter()
        # Processos novos a cada trial reinicializam os três blocos, os estados
        # de Adam, as estatísticas de BN, as máscaras Dropout e o embaralhamento.
        with tempfile.TemporaryDirectory(prefix=f"trial-{trial.trial_id}-", dir=workspace) as tmp:
            pasta = Path(tmp)
            (pasta / "dados").symlink_to(workspace / "particoes" / "dados", target_is_directory=True)
            shutil.copy2(workspace / "particoes" / "particoes.json", pasta / "particoes.json")
            subprocess.run(comando_treino(args, pasta, hp, somente_validacao=True),
                           cwd=ROOT, env=WORKER_ENV, check=True)
            resultado = json.loads((pasta / "resultados.json").read_text())
            if resultado["teste"] is not None or resultado["teste_clientes"]:
                raise AssertionError("Uma tentativa da busca acessou o teste.")
            score = resultado["melhor_val_loss"]
            if not math.isfinite(score):
                raise ValueError("Objetivo não finito.")
            resumo["trials"].append({"id": trial.trial_id, "hiperparametros": hp,
                "val_loss": score, "melhor_epoca": resultado["melhor_epoca"],
                "epocas_executadas": resultado["epocas_executadas"],
                "segundos": time.perf_counter() - inicio})
            salvar_json(output / "busca.json", resumo)
        # O Tuner recebe uma métrica do loop distribuído; não há model.fit local.
        return {"val_loss": score}


def executar(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Escolha um --output vazio para preservar os resultados.")
    output.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()
    resumo = {"status": "buscando", "algoritmo": "Keras Tuner RandomSearch",
        "keras_tuner": version("keras-tuner"), "objetivo": "min val_loss agregada",
        "max_trials": args.trials, "epocas_maximas": args.epocas, "patience": args.patience,
        "seed": args.seed, "clientes": 2, "batch_size": 64,
        "espaco": {"dropout": [0.0, 0.1, 0.2, 0.3, 0.4], "pooling_extra": [False, True],
                   "batch_normalization": [False, True], "learning_rate": [0.001, 0.0003]},
        "trials": []}
    salvar_json(output / "busca.json", resumo)
    try:
        with tempfile.TemporaryDirectory(prefix="lab5-busca-") as tmp:
            workspace = Path(tmp)
            subprocess.run([sys.executable, str(ROOT / "dados.py"),
                "--output", str(workspace / "particoes"), "--seed", str(args.seed), "--sem-teste"],
                cwd=ROOT, env=WORKER_ENV, check=True)
            resumo["particoes_busca"] = json.loads((workspace / "particoes" / "particoes.json").read_text())
            hp = kt.HyperParameters()
            hp.Choice("dropout", resumo["espaco"]["dropout"])
            hp.Boolean("pooling_extra")
            hp.Boolean("batch_normalization")
            hp.Choice("learning_rate", resumo["espaco"]["learning_rate"])
            tuner = BuscaDistribuida(hypermodel=None, objective=kt.Objective("val_loss", "min"),
                max_trials=args.trials, seed=args.seed, hyperparameters=hp,
                directory=str(workspace / "tuner"), project_name="u_shaped", overwrite=True,
                max_retries_per_trial=0, max_consecutive_failed_trials=1)
            tuner.search(args=args, workspace=workspace, output=output, resumo=resumo)
            if len(resumo["trials"]) != args.trials:
                raise RuntimeError("A busca terminou sem completar todas as configurações solicitadas.")
            best_trial = tuner.oracle.get_best_trials(1)[0]
            melhores = dict(best_trial.hyperparameters.values)
            resumo.update(status="treinando_final", melhor_trial=best_trial.trial_id,
                          melhores_hiperparametros=melhores, segundos_busca=time.perf_counter() - inicio)
            salvar_json(output / "busca.json", resumo)

        # A escolha já está fixada. Repetimos o treino do vencedor do zero, com
        # os mesmos dados/semente, e só então avaliamos o teste oficial uma vez.
        subprocess.run(comando_treino(args, output, melhores), cwd=ROOT, env=WORKER_ENV, check=True)
        resumo.update(status="concluido", segundos_total=time.perf_counter() - inicio)
        salvar_json(output / "busca.json", resumo)
        # Partições e logs de execução são reproduzíveis; a entrega retém os
        # resultados de todos os trials e apenas os modelos do treino final.
        shutil.rmtree(output / "dados")
        (output / "servidor.log").unlink(missing_ok=True)
        from relatorio import gerar
        gerar(output)
        print(f"Busca concluída: {args.trials} configurações. Resultado final em {output}", flush=True)
    except BaseException as exc:
        resumo.update(status="falhou", erro=str(exc), segundos_total=time.perf_counter() - inicio)
        salvar_json(output / "busca.json", resumo)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "nova_busca")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memoria-mb", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.trials <= 40 or args.epocas < 1 or args.patience < 1 or args.memoria_mb < 256:
        parser.error("Use trials entre 1 e 40, épocas/patience >= 1 e memória >= 256.")
    executar(args)
