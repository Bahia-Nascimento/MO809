"""Treino U-Shaped: clientes Ray independentes e um servidor gRPC compartilhado."""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import grpc
import ray
from config import ROOT, OUTPUT, MAX_BATCH, GRPC_OPTIONS, salvar_json
import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc


@ray.remote(num_cpus=1)
class ClienteActor:
    def __init__(self, codigo_root, **kwargs):
        # Caminhos com espaço funcionam sem alterar PYTHONPATH dos workers Ray.
        sys.path.insert(0, codigo_root)
        from cliente import Cliente
        self.cliente = Cliente(**kwargs)

    def info(self):
        return self.cliente.info()

    def iniciar_epoca(self):
        return self.cliente.iniciar_epoca()

    def treinar_lote(self, epoch, batch):
        return self.cliente.treinar_lote(epoch, batch)

    def avaliar(self, split):
        return self.cliente.avaliar(split)

    def salvar(self):
        return self.cliente.salvar()

    def restaurar(self):
        return self.cliente.restaurar()

    def fechar(self):
        return self.cliente.fechar()


def agregar(resultados):
    n = sum(r["n"] for r in resultados)
    return {"n": n, "loss": sum(r["loss"] * r["n"] for r in resultados) / n,
            "accuracy": sum(r["correct"] for r in resultados) / n,
            "tx_bytes": sum(r["tx_bytes"] for r in resultados),
            "rx_bytes": sum(r["rx_bytes"] for r in resultados)}


def esperar_servidor(process, ready_file, instance_id):
    limite = time.monotonic() + 90
    while time.monotonic() < limite:
        if process.poll() is not None:
            raise RuntimeError("Servidor terminou durante a inicialização; consulte servidor.log.")
        if ready_file.exists():
            ready = json.loads(ready_file.read_text())
            endereco = f"127.0.0.1:{ready['port']}"
            with grpc.insecure_channel(endereco, options=GRPC_OPTIONS) as channel:
                info = pb2_grpc.SplitLearningStub(channel).GetInfo(pb2.Empty(), timeout=10)
            if info.instance_id != instance_id:
                raise RuntimeError("Servidor de outra execução.")
            return endereco, info
        time.sleep(0.1)
    raise TimeoutError("Servidor não iniciou em 90 segundos.")


def executar(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError("Diretório já usado. Escolha outro --output para preservar o resultado.")
    manifest = out / "particoes.json"
    if not manifest.exists():
        subprocess.run([sys.executable, str(ROOT / "dados.py"), "--output", str(out),
            "--clientes", str(args.clientes), "--seed", str(args.seed),
            "--limite-treino", str(args.limite_treino), "--limite-teste", str(args.limite_teste)]
            + (["--sem-teste"] if args.somente_validacao else []), check=True)
    particoes = json.loads(manifest.read_text())
    if (particoes["seed"] != args.seed or len(particoes["clientes"]) != args.clientes
            or particoes.get("limite_treino", 0) != args.limite_treino
            or particoes.get("limite_teste", 0) != args.limite_teste):
        raise ValueError("Partições existentes incompatíveis com a configuração solicitada.")
    if not args.somente_validacao and not particoes.get("inclui_teste", True):
        raise ValueError("A avaliação final exige partições com teste.")
    instance_id = uuid.uuid4().hex
    configuracao = {**vars(args), "output": Path(os.path.relpath(out, ROOT)).as_posix(), "instance_id": instance_id,
        "inicio_utc": datetime.now(timezone.utc).isoformat(), "status": "executando",
        "versoes": {p: version(p) for p in ("tensorflow", "keras", "numpy", "grpcio", "protobuf", "ray")}}
    salvar_json(out / "config.json", configuracao)
    try:
        resultados = treinar(args, out, instance_id)
        salvar_json(out / "resultados.json", resultados)
        if not args.somente_validacao:
            from relatorio import gerar
            gerar(out)
    except BaseException as exc:
        configuracao.update(status="falhou", erro=str(exc))
        salvar_json(out / "config.json", configuracao)
        raise
    configuracao["status"] = "concluido"
    salvar_json(out / "config.json", configuracao)
    print(f"Concluído. Melhor val_loss: {resultados['melhor_val_loss']:.4f}. Resultados em {out}", flush=True)


def treinar(args, out, instance_id):
    inicio = time.perf_counter()
    history, best_loss, best_epoch = [], math.inf, 0
    with tempfile.TemporaryDirectory(prefix="lab5-") as tmp:
        ready_file = Path(tmp) / "ready.json"
        with (out / "servidor.log").open("w") as log:
            server = subprocess.Popen([sys.executable, "-u", str(ROOT / "servidor.py"),
                "--output", str(out), "--clientes", str(args.clientes), "--port", "0",
                "--seed", str(args.seed), "--memoria-mb", str(args.memoria_mb),
                "--learning-rate", str(args.learning_rate),
                "--ready-file", str(ready_file), "--instance-id", instance_id],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            channel = None
            try:
                endereco, server_info = esperar_servidor(server, ready_file, instance_id)
                if "GPU" not in server_info.device:
                    raise RuntimeError("M2 não está executando na GPU.")
                channel = grpc.insecure_channel(endereco, options=GRPC_OPTIONS)
                stub = pb2_grpc.SplitLearningStub(channel)
                ray.init(num_cpus=args.clientes, num_gpus=1, include_dashboard=False,
                         log_to_driver=True, object_store_memory=128 * 1024 * 1024)
                atores, infos = [], []
                for i in range(args.clientes):
                    ator = ClienteActor.options(num_gpus=1 / args.clientes).remote(
                        codigo_root=str(ROOT), client_id=i, endereco=endereco, output=str(out),
                        seed=args.seed, batch_size=args.batch_size, memoria_mb=args.memoria_mb,
                        learning_rate=args.learning_rate, dropout=args.dropout,
                        pooling_extra=args.pooling_extra, batch_normalization=args.batch_normalization,
                        somente_validacao=args.somente_validacao)
                    atores.append(ator)
                    infos.append(ray.get(ator.info.remote(), timeout=180))
                pids = [server_info.pid] + [info["pid"] for info in infos]
                if len(set(pids)) != len(pids) or any("GPU" not in i["device"] for i in infos):
                    raise RuntimeError("Exigimos processos distintos e forward na GPU em todos os blocos.")
                with (out / "historico.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = None
                    for epoch in range(1, args.epocas + 1):
                        epoch_start = time.perf_counter()
                        ray.get([a.iniciar_epoca.remote() for a in atores], timeout=60)
                        lotes = []
                        for batch in range(max(i["batches"] for i in infos)):
                            # Um lote completo por vez: M2 NÃO muda entre seu forward
                            # e backward. Round-robin permite colaboração sem tapes
                            # obsoletos, e alterna quem começa a cada época.
                            for offset in range(args.clientes):
                                i = (offset + epoch - 1) % args.clientes
                                if batch < infos[i]["batches"]:
                                    lotes.append(ray.get(atores[i].treinar_lote.remote(epoch, batch), timeout=120))
                            if batch % 100 == 0:
                                print(f"Época {epoch}/{args.epocas}, lote local {batch + 1}", flush=True)
                        validacao = [ray.get(a.avaliar.remote("val"), timeout=180) for a in atores]
                        train, val = agregar(lotes), agregar(validacao)
                        if not math.isfinite(train["loss"]) or not math.isfinite(val["loss"]):
                            raise ValueError("Loss não finita durante o treinamento.")
                        if val["loss"] < best_loss:
                            # Um checkpoint é o CONJUNTO consistente: M2 e M1/M3
                            # de cada cliente, todos na mesma fronteira de época.
                            stub.SaveCheckpoint(pb2.Empty(), timeout=60)
                            for a in atores:
                                ray.get(a.salvar.remote(), timeout=60)
                            best_loss, best_epoch = val["loss"], epoch
                        row = {"epoch": epoch, "train_loss": train["loss"],
                            "train_accuracy": train["accuracy"], "val_loss": val["loss"],
                            "val_accuracy": val["accuracy"], "train_n": train["n"],
                            "val_n": val["n"], "tx_bytes": train["tx_bytes"], "rx_bytes": train["rx_bytes"],
                            "rpc_seconds": sum(r["rpc_seconds"] for r in lotes),
                            "seconds": time.perf_counter() - epoch_start}
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        f.flush()
                        history.append({**row, "validacao_clientes": validacao})
                        print(f"Época {epoch}: loss={train['loss']:.4f}, acc={train['accuracy']:.2%}, "
                              f"val_loss={val['loss']:.4f}, val_acc={val['accuracy']:.2%}", flush=True)
                        # A decisão é única para todo o conjunto de modelos;
                        # callbacks separados por cliente romperiam a sincronia.
                        if args.patience and epoch - best_epoch >= args.patience:
                            print(f"Early stopping: {args.patience} épocas sem melhora.", flush=True)
                            break
                # O teste oficial só é acessado depois da seleção pela validação.
                updates = stub.LoadCheckpoint(pb2.Empty(), timeout=60).updates
                for a in atores:
                    ray.get(a.restaurar.remote(), timeout=60)
                validacao_restaurada = agregar([
                    ray.get(a.avaliar.remote("val"), timeout=180) for a in atores])
                if abs(validacao_restaurada["loss"] - best_loss) > 1e-5:
                    raise AssertionError("Checkpoint conjunto não reproduziu a melhor validação.")
                testes = ([] if args.somente_validacao else
                    [ray.get(a.avaliar.remote("test"), timeout=180) for a in atores])
                for a in atores:
                    ray.get(a.fechar.remote(), timeout=30)
                esperado = len(history) * sum(i["batches"] for i in infos)
                if updates != esperado:
                    raise AssertionError(f"Número de updates {updates} != {esperado}.")
                return {"melhor_epoca": best_epoch, "melhor_val_loss": best_loss, "epocas": history,
                    "teste": agregar(testes) if testes else None, "teste_clientes": testes,
                    "validacao_restaurada": validacao_restaurada, "updates_m2": updates,
                    "epocas_executadas": len(history), "parada_antecipada": len(history) < args.epocas,
                    "dispositivos": {"servidor": {"pid": server_info.pid, "device": server_info.device},
                                     "clientes": infos},
                    "segundos": time.perf_counter() - inicio}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--clientes", type=int, default=2)
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling-extra", action="store_true")
    parser.add_argument("--batch-normalization", action="store_true")
    parser.add_argument("--patience", type=int, default=0, help="Épocas sem melhora; 0 desativa a parada.")
    parser.add_argument("--somente-validacao", action="store_true", help="Não disponibiliza nem avalia teste.")
    parser.add_argument("--memoria-mb", type=int, default=1024)
    parser.add_argument("--limite-treino", type=int, default=0, help="Máximo de imagens de treino; 0 usa todas.")
    parser.add_argument("--limite-teste", type=int, default=0)
    args = parser.parse_args()
    if (args.clientes < 1 or args.epocas < 1 or not 1 <= args.batch_size <= MAX_BATCH
            or args.learning_rate <= 0 or not math.isfinite(args.learning_rate)
            or not 0 <= args.dropout < 1 or args.patience < 0
            or args.memoria_mb < 256 or min(args.limite_treino, args.limite_teste) < 0):
        parser.error("Parâmetros inválidos: clientes/épocas >=1, batch 1..256, LR>0, memória>=256, limites>=0.")
    executar(args)
