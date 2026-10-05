"""Servidor gRPC mantém apenas M2 e o tape de um único lote em andamento."""
import argparse
from concurrent import futures
import os
from pathlib import Path
import threading
import time
import uuid

import grpc
import numpy as np
from config import GRPC_OPTIONS, MAX_BATCH, configurar_tensorflow, salvar_json
import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc


class SplitLearningService(pb2_grpc.SplitLearningServicer):
    def __init__(self, clientes=2, output=None, instance_id=None, timeout=60, learning_rate=0.001):
        import tensorflow as tf
        from modelo import create_server_model
        self.tf = tf
        self.server_model = create_server_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.clientes = clientes
        self.output = Path(output) if output else None
        self.instance_id = instance_id or uuid.uuid4().hex
        self.timeout = timeout
        self.lock = threading.Lock()
        self.pending = None
        self.failed = False
        self.restored = False
        self.next_step = [0] * clientes
        self.updates = 0
        # Evidência do dispositivo que realmente executa o forward.
        self.device = self.server_model(tf.zeros((1, 128))).device

    def _check(self, context):
        if self.pending and time.monotonic() - self.pending["created"] > self.timeout:
            self.pending = None  # Libera o tape e sua memória; não reaproveita a sessão.
            self.failed = True
        if self.failed:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Sessão interrompida: reinicie o experimento; não há retomada parcial.")

    def _input(self, request, width, context):
        if not 0 <= request.client_id < self.clientes:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Cliente desconhecido.")
        if not 1 <= request.batch_size <= MAX_BATCH:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Tamanho de lote inválido.")
        values = np.asarray(request.values, dtype=np.float32)
        if values.size != request.batch_size * width or not np.isfinite(values).all():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Dimensão inválida ou NaN/Inf.")
        return self.tf.convert_to_tensor(values.reshape(request.batch_size, width))

    def _info(self):
        return pb2.Info(instance_id=self.instance_id, updates=self.updates, pid=os.getpid(),
                        device=self.device, clients=self.clientes)

    def GetInfo(self, request, context):
        with self.lock:
            self._check(context)
            return self._info()

    def Forward(self, request, context):
        with self.lock:
            self._check(context)
            if self.restored:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Checkpoint restaurado para inferência.")
            activations = self._input(request, 128, context)
            if self.pending:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Outro lote aguarda Backward.")
            if request.step != self.next_step[request.client_id] or request.token:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Passo inesperado ou repetido.")
            # O tape continua vivo entre DOIS RPCs. watch é necessário porque a
            # ativação recebida é um Tensor, não uma variável treinável de M2.
            with self.tf.GradientTape() as tape:
                tape.watch(activations)
                outputs = self.server_model(activations, training=True)
            self.tf.debugging.assert_all_finite(outputs, "Ativações de M2 não finitas.")
            token = uuid.uuid4().hex
            self.pending = dict(client=request.client_id, step=request.step,
                batch=request.batch_size, token=token, tape=tape, inputs=activations,
                outputs=outputs, created=time.monotonic())
            return pb2.ServerToClient(values=outputs.numpy().ravel(), batch_size=request.batch_size,
                                      step=request.step, token=token)

    def Backward(self, request, context):
        with self.lock:
            self._check(context)
            gradients = self._input(request, 64, context)
            p = self.pending
            if not p or (request.client_id, request.step, request.batch_size, request.token) != (
                    p["client"], p["step"], p["batch"], p["token"]):
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Backward não corresponde ao Forward.")
            from modelo import verificar_gradientes
            try:
                # Regra da cadeia: (d M2 / d entrada)^T * (d loss / d saída M2).
                # Uma única chamada obtém gradientes de entrada E pesos; não há
                # necessidade de persistent=True, que reteria memória extra.
                grads = p["tape"].gradient(p["outputs"],
                    [p["inputs"]] + self.server_model.trainable_variables,
                    output_gradients=gradients)
                verificar_gradientes(grads)
                response = pb2.ServerToClient(values=grads[0].numpy().ravel(),
                    batch_size=p["batch"], step=p["step"], token=p["token"])
                self.optimizer.apply_gradients(zip(grads[1:], self.server_model.trainable_variables))
            except Exception:
                self.failed = True
                raise
            finally:
                self.pending = None
            self.next_step[request.client_id] += 1
            self.updates += 1
            return response

    def Evaluate(self, request, context):
        with self.lock:
            self._check(context)
            activations = self._input(request, 128, context)
            if self.pending:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Conclua o lote antes de avaliar.")
            outputs = self.server_model(activations, training=False)
            return pb2.ServerToClient(values=outputs.numpy().ravel(), batch_size=request.batch_size,
                                      step=request.step)

    def _checkpoint(self, context, load=False):
        self._check(context)
        if self.pending or self.output is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Checkpoint indisponível.")
        arquivo = self.output / "M2.keras"
        if load:
            if not arquivo.exists():
                context.abort(grpc.StatusCode.NOT_FOUND, "Checkpoint ausente.")
            # Restaura somente pesos para inferência final. Retomar treinamento
            # exigiria também Adam, passos e os estados locais de TODOS os clientes.
            self.server_model.set_weights(self.tf.keras.models.load_model(arquivo).get_weights())
            self.restored = True
        else:
            self.server_model.save(arquivo)
        return self._info()

    def SaveCheckpoint(self, request, context):
        with self.lock:
            return self._checkpoint(context)

    def LoadCheckpoint(self, request, context):
        with self.lock:
            return self._checkpoint(context, load=True)


def serve(args):
    configurar_tensorflow(gpu=not args.cpu, seed=args.seed, memoria_mb=args.memoria_mb)
    args.output.mkdir(parents=True, exist_ok=True)
    service = SplitLearningService(args.clientes, args.output, args.instance_id,
                                  learning_rate=args.learning_rate)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_OPTIONS)
    pb2_grpc.add_SplitLearningServicer_to_server(service, server)
    port = server.add_insecure_port(f"127.0.0.1:{args.port}")
    if not port:
        raise RuntimeError("Não foi possível abrir a porta gRPC.")
    server.start()
    if args.ready_file:
        # Publicação atômica impede ler JSON parcialmente escrito.
        temp = args.ready_file.with_suffix(".tmp")
        salvar_json(temp, {"port": port, "instance_id": service.instance_id})
        temp.replace(args.ready_file)
    print(f"Servidor PID={os.getpid()}, porta={port}, dispositivo={service.device}", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(2).wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clientes", type=int, default=2)
    parser.add_argument("--port", type=int, default=50055)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--memoria-mb", type=int, default=1024)
    parser.add_argument("--instance-id")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--cpu", action="store_true", help="Somente diagnóstico/testes.")
    serve(parser.parse_args())
