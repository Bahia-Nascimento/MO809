"""Ray inicia dois clientes independentes; gRPC transporta os vetores ao servidor.

Ray é o orquestrador de processos, não substitui o RPC solicitado no exercício.
O servidor é um subprocesso separado e escolhe uma porta livre automaticamente.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import grpc
import numpy as np
import ray
from config import ROOT, OUTPUT, salvar_json
import ml_pb2
import ml_pb2_grpc


@ray.remote(num_cpus=1)
class ClienteActor:
    def __init__(self, client_id, endereco, output, seed, codigo_root):
        # Importa o código mesmo se o usuário iniciar o script de outra pasta.
        # Não exige runtime_env nem comandos shell com caminhos contendo espaços.
        sys.path.insert(0, codigo_root)
        from mlcliente import Cliente
        self.cliente = Cliente(client_id, endereco, Path(output), seed)

    def executar(self, rodada):
        return self.cliente.executar(rodada)


def esperar_servidor(process, ready_file, instance_id):
    limite = time.monotonic() + 60
    while time.monotonic() < limite:
        if process.poll() is not None:
            raise RuntimeError("Servidor terminou antes de iniciar. Consulte output/servidor.log.")
        if ready_file.exists():
            ready = json.loads(ready_file.read_text())
            endereco = f"127.0.0.1:{ready['port']}"
            with grpc.insecure_channel(endereco) as channel:
                info = ml_pb2_grpc.SimilaridadeStub(channel).GetInfo(ml_pb2.Vazio(), timeout=5)
            if info.instance_id != instance_id:
                raise RuntimeError("A porta respondeu com outra instância de servidor.")
            return endereco
        time.sleep(0.1)
    raise TimeoutError("Servidor não iniciou em 60 segundos.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rodadas", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rodadas < 1:
        parser.error("Rodadas deve ser >= 1.")
    out = args.output.resolve()
    if not (out / "metadata.json").exists():
        parser.error("Execute treinar.py antes do experimento.")
    resultados = []
    instance_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="lab4-") as tmp, (out / "servidor.log").open("w") as log:
        ready_file = Path(tmp) / "ready.json"
        server = subprocess.Popen([sys.executable, "-u", str(ROOT / "mlservidor.py"),
            "--output", str(out), "--port", "0", "--ready-file", str(ready_file),
            "--instance-id", instance_id], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            endereco = esperar_servidor(server, ready_file, instance_id)
            ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
                     object_store_memory=100 * 1024 * 1024)
            atores = [ClienteActor.remote(i, endereco, str(out), args.seed, str(ROOT)) for i in range(2)]
            for i in range(args.rodadas):
                # Dispara AMBOS antes de ray.get: o primeiro RPC espera o segundo.
                futuros = [ator.executar.remote(f"{instance_id}-{i}") for ator in atores]
                respostas = ray.get(futuros, timeout=120)
                a, b = respostas
                if a["pid"] == b["pid"]:
                    raise AssertionError("Clientes deveriam executar em processos distintos.")
                # Diagnóstico: reconstrói a conta com os vetores retornados ao orquestrador.
                # Isso não participa da decisão do servidor, apenas valida o transporte.
                va, vb = np.asarray(a["vetor"], dtype="float32"), np.asarray(b["vetor"], dtype="float32")
                local = float(np.sqrt(max(float(np.sum((va-vb)**2)), 1e-7)))
                np.testing.assert_allclose(a["distancia"], local, atol=1e-6)
                np.testing.assert_allclose(a["distancia"], b["distancia"], atol=1e-7)
                real = a["classe"] == b["classe"]
                resultados.append({"rodada": i, "clientes": respostas, "distancia_local": local,
                                   "similar_real": real, "acerto": a["similares"] == real})
                print(f"Rodada {i}: classes {a['classe']}/{b['classe']}, D={a['distancia']:.4f}, "
                      f"similares={a['similares']}, acerto={a['similares'] == real}", flush=True)
        finally:
            ray.shutdown()
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
    salvar_json(out / "experimento.json", resultados)
    linhas = ["# Demonstração distribuída", "", "Dois processos Ray calculam embeddings; o servidor compara via gRPC.",
              "Esta amostra pequena e aleatória demonstra a integração; a avaliação principal usa pares de teste balanceados.",
              "", "| Rodada | Classes | Distância | Predição | Acerto |", "|---|---|---:|---|---|"]
    for row in resultados:
        a,b = row["clientes"]
        linhas.append(f"| {row['rodada']} | {a['classe']} / {b['classe']} | {a['distancia']:.4f} | "
                      f"{'similar' if a['similares'] else 'diferente'} | {row['acerto']} |")
    (out / "experimento.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"Resultados em {out / 'experimento.md'}")


if __name__ == "__main__":
    main()
