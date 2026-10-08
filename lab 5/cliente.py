"""Cliente possui os dados, M1, M3, a loss e dois otimizadores locais."""
import csv
import math
import os
from pathlib import Path
import time

import grpc
import numpy as np
from config import GRPC_OPTIONS, configurar_tensorflow
import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc


def get_activations(model, x):
    import tensorflow as tf
    with tf.GradientTape() as tape:
        activations = model(x, training=True)
    return activations, tape


def mensagem(values, client_id, step, token=""):
    return pb2.ClientToServer(values=values.numpy().ravel(), batch_size=int(values.shape[0]),
                              client_id=client_id, step=step, token=token)


def tensor_resposta(response, batch, width, step, token=None):
    import tensorflow as tf
    values = np.asarray(response.values, dtype=np.float32)
    if response.batch_size != batch or response.step != step or values.size != batch * width:
        raise ValueError("Resposta incompatível com o lote enviado.")
    if not np.isfinite(values).all() or (token is not None and response.token != token):
        raise ValueError("Resposta não finita ou token inesperado.")
    return tf.convert_to_tensor(values.reshape(batch, width))


def train_step(model, head, x_batch, y_batch, optimizer, head_optimizer, stub, client_id, step):
    import tensorflow as tf
    from modelo import verificar_gradientes
    inicio = time.perf_counter()
    # Primeiro corte: o cliente retém o tape de M1 enquanto M2 executa remotamente.
    a, tape1 = get_activations(model, x_batch)
    forward = mensagem(a, client_id, step)
    inicio_rpc = time.perf_counter()
    response = stub.Forward(forward, timeout=60)
    rpc_seconds = time.perf_counter() - inicio_rpc
    if not response.token:
        raise ValueError("Servidor não identificou o Forward.")
    b = tensor_resposta(response, len(x_batch), 64, step)
    with tf.GradientTape() as tape3:
        tape3.watch(b)
        predictions = head(b, training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions))
    # A loss média já contém 1/batch. Não divida novamente os gradientes recebidos.
    grads3 = tape3.gradient(loss, [b] + head.trainable_variables)
    verificar_gradientes(grads3)
    backward = mensagem(grads3[0], client_id, step, response.token)
    inicio_rpc = time.perf_counter()
    returned = stub.Backward(backward, timeout=60)
    rpc_seconds += time.perf_counter() - inicio_rpc
    da = tensor_resposta(returned, len(x_batch), 128, step, response.token)
    grads1 = tape1.gradient(a, model.trainable_variables, output_gradients=da)
    verificar_gradientes(grads1)
    # Os gradientes dos três blocos referem-se ao mesmo forward. Atualizamos os
    # blocos locais após a resposta; o orquestrador só então libera o próximo lote.
    optimizer.apply_gradients(zip(grads1, model.trainable_variables))
    head_optimizer.apply_gradients(zip(grads3[1:], head.trainable_variables))
    correct = tf.reduce_sum(tf.cast(tf.argmax(predictions, axis=1, output_type=tf.int64)
                                   == tf.cast(y_batch, tf.int64), tf.int32))
    return {"n": len(x_batch), "loss": float(loss), "correct": int(correct),
            "accuracy": float(correct) / len(x_batch), "rpc_seconds": rpc_seconds,
            "batch_seconds": time.perf_counter() - inicio,
            # ByteSize inclui TODOS os campos Protobuf dos dois RPCs, mas não
            # cabeçalhos gRPC/HTTP2/TCP. Não é tráfego capturado na interface de rede.
            "tx_bytes": forward.ByteSize() + backward.ByteSize(),
            "rx_bytes": response.ByteSize() + returned.ByteSize()}


class Cliente:
    def __init__(self, client_id, endereco, output, seed=42, batch_size=64,
                 memoria_mb=1024, gpu=True, learning_rate=0.001, dropout=0.0,
                 pooling_extra=False, batch_normalization=False, somente_validacao=False):
        self.tf = configurar_tensorflow(gpu=gpu, seed=seed, memoria_mb=memoria_mb)
        from modelo import create_partial_model, create_head_model
        self.client_id = client_id
        self.output = Path(output)
        self.folder = self.output / f"cliente_{client_id}"
        self.folder.mkdir(exist_ok=True)
        # Cada processo lê SOMENTE seu arquivo local; imagens/rótulos não vão a M2.
        with np.load(self.output / "dados" / f"cliente_{client_id}.npz") as data:
            splits = ("train", "val") if somente_validacao else ("train", "val", "test")
            self.data = {f"{k}_{s}": data[f"{k}_{s}"] for s in splits for k in ("x", "y")}
        self.partial_model = create_partial_model(dropout=dropout, pooling_extra=pooling_extra,
                                                  batch_normalization=batch_normalization)
        self.head_model = create_head_model()
        self.optimizer = self.tf.keras.optimizers.Adam(learning_rate)
        self.head_optimizer = self.tf.keras.optimizers.Adam(learning_rate)
        self.channel = grpc.insecure_channel(endereco, options=GRPC_OPTIONS)
        self.stub = pb2_grpc.SplitLearningStub(self.channel)
        self.rng = np.random.default_rng(seed + client_id)
        self.batch_size = batch_size
        self.step = 0
        self.order = None
        self.restored = False
        self.device = self.partial_model(self.tf.zeros((1, 32, 32, 3))).device
        self.log = (self.folder / "results.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.log, fieldnames=["epoch", "batch", "client_id", "step",
            "n", "loss", "correct", "accuracy", "rpc_seconds", "batch_seconds", "tx_bytes", "rx_bytes"])
        self.writer.writeheader()

    def info(self):
        return {"client_id": self.client_id, "pid": os.getpid(), "device": self.device,
                "train_size": len(self.data["x_train"]),
                "batches": math.ceil(len(self.data["x_train"]) / self.batch_size)}

    def iniciar_epoca(self):
        self.order = self.rng.permutation(len(self.data["x_train"]))

    def treinar_lote(self, epoch, batch):
        if self.restored:
            raise RuntimeError("Checkpoint restaurado apenas para inferência.")
        indices = self.order[batch * self.batch_size:(batch + 1) * self.batch_size]
        # Conservar imagens em uint8 economiza RAM; normaliza somente o lote atual.
        x = self.data["x_train"][indices].astype(np.float32) / 255.0
        y = self.data["y_train"][indices]
        result = train_step(self.partial_model, self.head_model, x, y, self.optimizer,
                            self.head_optimizer, self.stub, self.client_id, self.step)
        result.update(epoch=epoch, batch=batch, client_id=self.client_id, step=self.step)
        self.writer.writerow(result)
        self.log.flush()
        self.step += 1
        return result

    def avaliar(self, split="val"):
        if split not in ("val", "test"):
            raise ValueError("Use val ou test.")
        if f"x_{split}" not in self.data:
            raise ValueError("Conjunto de teste indisponível durante a busca.")
        x, y = self.data[f"x_{split}"], self.data[f"y_{split}"]
        loss_sum, correct, tx, rx = 0.0, 0, 0, 0
        for start in range(0, len(x), self.batch_size):
            xb = x[start:start + self.batch_size].astype(np.float32) / 255.0
            yb = y[start:start + self.batch_size]
            a = self.partial_model(xb, training=False)
            request = mensagem(a, self.client_id, self.step)
            response = self.stub.Evaluate(request, timeout=60)
            b = tensor_resposta(response, len(xb), 64, self.step)
            predictions = self.head_model(b, training=False)
            losses = self.tf.keras.losses.sparse_categorical_crossentropy(yb, predictions)
            loss_sum += float(self.tf.reduce_sum(losses))
            correct += int(np.sum(np.argmax(predictions.numpy(), axis=1) == yb))
            tx += request.ByteSize()
            rx += response.ByteSize()
        return {"client_id": self.client_id, "n": len(x), "loss": loss_sum / len(x),
                "correct": correct, "accuracy": correct / len(x), "tx_bytes": tx, "rx_bytes": rx}

    def salvar(self):
        self.partial_model.save(self.folder / "M1.keras")
        self.head_model.save(self.folder / "M3.keras")

    def restaurar(self):
        self.partial_model.set_weights(self.tf.keras.models.load_model(self.folder / "M1.keras").get_weights())
        self.head_model.set_weights(self.tf.keras.models.load_model(self.folder / "M3.keras").get_weights())
        self.restored = True

    def fechar(self):
        self.log.close()
        self.channel.close()
