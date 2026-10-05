"""Recarrega checkpoints em novos processos Ray/gRPC e confere o teste.

A auditoria reproduz o caminho distribuído. Diretórios temporários evitam
sobrescrever os CSVs originais ao criar os clientes de avaliação.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import grpc
import numpy as np
import ray
from config import ROOT, OUTPUT, GRPC_OPTIONS, salvar_json
from experimento import ClienteActor, esperar_servidor
import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc


def verificar(output):
    out = Path(output).resolve()
    config = json.loads((out / "config.json").read_text())
    manifesto = json.loads((out / "particoes.json").read_text())
    esperado = json.loads((out / "resultados.json").read_text())
    n_clientes = config["clientes"]
    if not len(manifesto["clientes"]) == len(esperado["teste_clientes"]) == n_clientes:
        raise AssertionError("Número de clientes inconsistente nos artefatos.")
    indices_treino_val, indices_teste = [], []
    for registro in manifesto["clientes"]:
        arquivo = out / "dados" / f"cliente_{registro['id']}.npz"
        if hashlib.sha256(arquivo.read_bytes()).hexdigest() != registro["sha256"]:
            raise AssertionError("Os dados foram alterados após o treinamento.")
        with np.load(arquivo) as data:
            indices_treino_val.extend([data["train_indices"], data["val_indices"]])
            indices_teste.append(data["test_indices"])
    for indices in (indices_treino_val, indices_teste):
        unidos = np.concatenate(indices)
        if len(np.unique(unidos)) != len(unidos):
            raise AssertionError("Sobreposição de partições.")

    resultados = []
    instance_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="lab5-verificacao-") as tmp:
        temp = Path(tmp)
        (temp / "dados").mkdir()
        for i in range(n_clientes):
            # Hard links não duplicam os arquivos .npz; clientes apenas os leem.
            os.link(out / "dados" / f"cliente_{i}.npz", temp / "dados" / f"cliente_{i}.npz")
            (temp / f"cliente_{i}").mkdir()
            for name in ("M1.keras", "M3.keras"):
                shutil.copyfile(out / f"cliente_{i}" / name, temp / f"cliente_{i}" / name)
        with (out / "verificacao_servidor.log").open("w") as log:
            server = subprocess.Popen([sys.executable, str(ROOT / "servidor.py"),
                "--output", str(out), "--clientes", str(n_clientes), "--port", "0",
                "--seed", str(config["seed"]), "--memoria-mb", str(config["memoria_mb"]),
                "--ready-file", str(temp / "ready.json"), "--instance-id", instance_id],
                stdout=log, stderr=subprocess.STDOUT)
            channel = None
            try:
                endereco, info = esperar_servidor(server, temp / "ready.json", instance_id)
                if "GPU" not in info.device:
                    raise AssertionError("Servidor de avaliação não está na GPU.")
                channel = grpc.insecure_channel(endereco, options=GRPC_OPTIONS)
                stub = pb2_grpc.SplitLearningStub(channel)
                stub.LoadCheckpoint(pb2.Empty(), timeout=60)
                ray.init(num_cpus=n_clientes, num_gpus=1, include_dashboard=False,
                         log_to_driver=False, object_store_memory=128 * 1024 * 1024)
                atores = []
                for i in range(n_clientes):
                    ator = ClienteActor.options(num_gpus=1 / n_clientes).remote(
                        codigo_root=str(ROOT), client_id=i, endereco=endereco, output=str(temp),
                        seed=config["seed"], batch_size=config["batch_size"],
                        memoria_mb=config["memoria_mb"])
                    if "GPU" not in ray.get(ator.info.remote(), timeout=180)["device"]:
                        raise AssertionError("Cliente de avaliação não está na GPU.")
                    ray.get(ator.restaurar.remote(), timeout=60)
                    atores.append(ator)
                for i, ator in enumerate(atores):
                    actual = ray.get(ator.avaliar.remote("test"), timeout=180)
                    reference = esperado["teste_clientes"][i]
                    if reference["client_id"] != i or actual["correct"] != reference["correct"]:
                        raise AssertionError("Checkpoint não reproduz os acertos do teste original.")
                    np.testing.assert_allclose(actual["loss"], reference["loss"], atol=2e-6, rtol=2e-6)
                    resultados.append({**actual, "delta_loss": actual["loss"] - reference["loss"]})
                    ray.get(ator.fechar.remote(), timeout=30)
                if stub.GetInfo(pb2.Empty(), timeout=10).updates != 0:
                    raise AssertionError("A avaliação não pode atualizar M2.")
            finally:
                if channel:
                    channel.close()
                ray.shutdown()
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    result = {"status": "ok", "particoes_disjuntas": True, "hashes_conferidos": True,
              "checkpoint_reproduz_teste_grpc": True, "clientes": resultados}
    salvar_json(out / "verificacao_checkpoint.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    verificar(parser.parse_args().output)
