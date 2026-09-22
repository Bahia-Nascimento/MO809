"""Cliente: carrega o modelo, escolhe uma imagem e envia SOMENTE seu embedding."""
import argparse
import os
from pathlib import Path

import grpc
import numpy as np
from config import OUTPUT, configurar_tensorflow, hash_modelo, ler_metadata
import ml_pb2
import ml_pb2_grpc


class Cliente:
    def __init__(self, client_id, endereco="127.0.0.1:50054", output=OUTPUT, seed=42):
        if client_id not in (0, 1):
            raise ValueError("Use client_id 0 ou 1.")
        self.client_id = client_id
        self.rng = np.random.default_rng(seed + client_id)
        tf = configurar_tensorflow(gpu=False, seed=seed + client_id)
        self.meta = ler_metadata(output)
        self.model_id = hash_modelo(Path(output) / "rede_base.keras")
        if self.model_id != self.meta["model_id"]:
            raise ValueError("Arquivo de pesos não corresponde aos metadados.")
        # Cada processo tem sua PRÓPRIA cópia dos mesmos pesos, carregada uma vez.
        self.rede = tf.keras.models.load_model(Path(output) / "rede_base.keras", compile=False)
        with np.load(Path(output) / "dados_clientes.npz") as dados:
            self.imagens, self.labels = dados["imagens"], dados["labels"]
        self.channel = grpc.insecure_channel(endereco)
        self.stub = ml_pb2_grpc.SimilaridadeStub(self.channel)
        info = self.stub.GetInfo(ml_pb2.Vazio(), timeout=30)
        if info.model_id != self.model_id or info.dimensao != self.rede.output_shape[-1]:
            self.channel.close()
            raise ValueError("Modelo do cliente difere do esperado pelo servidor.")

    def executar(self, rodada):
        indice = int(self.rng.integers(len(self.imagens)))
        # (28,28) -> (1,28,28,1): um lote com uma imagem e um canal de cinza.
        imagem = self.imagens[indice].astype("float32")[None, ..., None] / 255.0
        vetor = self.rede([imagem], training=False).numpy()[0]
        resposta = self.stub.EnviarVetor(ml_pb2.VetorRequest(client_id=str(self.client_id),
            rodada=rodada, model_id=self.model_id, valores=vetor.tolist()), timeout=60)
        # Índice/rótulo abaixo são diagnóstico LOCAL; não constam na mensagem RPC.
        return {"client_id": self.client_id, "pid": os.getpid(), "indice": indice,
                "classe": int(self.labels[indice]), "vetor": vetor.tolist(), "rodada": resposta.rodada,
                "distancia": resposta.distancia, "similares": resposta.similares, "limiar": resposta.limiar}

    def fechar(self):
        self.channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client_id", type=int, choices=[0, 1])
    parser.add_argument("--endereco", default="127.0.0.1:50054")
    parser.add_argument("--rodada", default="manual-1", help="Use o mesmo nome nos dois clientes; novo nome para novo par.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cliente = Cliente(args.client_id, args.endereco, args.output, args.seed)
    try:
        print(cliente.executar(args.rodada))
    finally:
        cliente.fechar()


if __name__ == "__main__":
    main()
