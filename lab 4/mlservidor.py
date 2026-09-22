"""Servidor gRPC: recebe dois embeddings e executa a camada Lambda de distância.

Não treina e não recebe imagens nem rótulos. O estado de cada rodada é protegido
por Condition, porque o gRPC atende clientes em threads simultâneas.
"""
import argparse
from concurrent import futures
from pathlib import Path
import threading
import time
import uuid

import grpc
import numpy as np
from config import OUTPUT, configurar_tensorflow, hash_modelo, ler_metadata, salvar_json
import ml_pb2
import ml_pb2_grpc


class ServicoSimilaridade(ml_pb2_grpc.SimilaridadeServicer):
    def __init__(self, output=OUTPUT, timeout=30.0, instance_id=None):
        tf = configurar_tensorflow(gpu=False)
        import modelo  # Registra a função nomeada da Lambda antes de carregar o arquivo.
        self.meta = ler_metadata(output)
        if hash_modelo(Path(output) / "rede_base.keras") != self.meta["model_id"]:
            raise ValueError("O modelo base foi alterado depois da calibração.")
        self.comparador = tf.keras.models.load_model(Path(output) / "comparador.keras", compile=False)
        self.timeout = timeout
        self.instance_id = instance_id or uuid.uuid4().hex
        self.condition = threading.Condition()
        self.rodadas = {}

    def GetInfo(self, request, context):
        # Além de readiness, informa a versão dos pesos e a dimensão esperada.
        return ml_pb2.Info(model_id=self.meta["model_id"], dimensao=self.meta["dimensao"],
                           limiar=self.meta["limiar"], instance_id=self.instance_id)

    def EnviarVetor(self, request, context):
        if request.client_id not in ("0", "1"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Use client_id 0 ou 1.")
        if not request.rodada or len(request.rodada) > 128:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Rodada deve ter 1 a 128 caracteres.")
        if request.model_id != self.meta["model_id"]:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Clientes devem usar os mesmos pesos do servidor.")
        vetor = np.asarray(request.valores, dtype="float32")
        if vetor.shape != (self.meta["dimensao"],) or not np.isfinite(vetor).all():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Dimensão inválida ou valores NaN/Inf.")
        # Evita overflow de float32 ao elevar ao quadrado valores malformados.
        if np.max(np.abs(vetor)) > 1e10:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Magnitude do vetor inválida.")
        with self.condition:
            agora = time.monotonic()
            # Guarda resultados por 5 minutos para permitir repetição idempotente.
            for key in list(self.rodadas):
                estado = self.rodadas[key]
                if agora - estado["inicio"] > 300 and (estado["resultado"] is not None or estado["erro"]):
                    del self.rodadas[key]
            if request.rodada not in self.rodadas:
                if len(self.rodadas) >= 1024:
                    context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Limite de rodadas; aguarde a limpeza.")
                self.rodadas[request.rodada] = {"inicio": agora, "vetores": {}, "resultado": None, "erro": None}
            estado = self.rodadas[request.rodada]
            anterior = estado["vetores"].get(request.client_id)
            if anterior is not None and not np.array_equal(anterior, vetor):
                context.abort(grpc.StatusCode.ALREADY_EXISTS, "Cliente já enviou outro vetor nesta rodada.")
            estado["vetores"][request.client_id] = vetor
            if len(estado["vetores"]) == 2 and estado["resultado"] is None and not estado["erro"]:
                # Mesmo comparador salvo no treinamento: não redefinimos a métrica.
                a, b = [estado["vetores"][str(i)][None, :] for i in range(2)]
                distancia = float(self.comparador([a, b], training=False).numpy()[0, 0])
                estado["resultado"] = ml_pb2.Resultado(rodada=request.rodada, distancia=distancia,
                    similares=distancia <= self.meta["limiar"], limiar=self.meta["limiar"], clientes=["0", "1"])
                print(f"Rodada {request.rodada}: distância={distancia:.6f}, similares={estado['resultado'].similares}", flush=True)
                self.condition.notify_all()
            while estado["resultado"] is None and not estado["erro"]:
                restante = self.timeout - (time.monotonic() - estado["inicio"])
                if restante <= 0 or not context.is_active():
                    estado["erro"] = "Rodada interrompida: faltou um cliente ou a chamada foi cancelada. Use nova rodada."
                    self.condition.notify_all()
                    break
                # wait libera o lock: a segunda thread precisa entrar para completar o par.
                self.condition.wait(timeout=min(restante, 0.2))
            if estado["erro"]:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, estado["erro"])
            return estado["resultado"]


def criar_servidor(output=OUTPUT, endereco="127.0.0.1:50054", timeout=30, instance_id=None):
    servico = ServicoSimilaridade(output, timeout, instance_id)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8),
                         options=[("grpc.so_reuseport", 0)])
    ml_pb2_grpc.add_SimilaridadeServicer_to_server(servico, server)
    porta = server.add_insecure_port(endereco)
    if not porta:
        raise RuntimeError(f"Não foi possível usar {endereco}")
    server.start()
    return server, porta, servico.instance_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50054)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--instance-id")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("Timeout deve ser positivo.")
    server, port, instance_id = criar_servidor(args.output, f"{args.host}:{args.port}", args.timeout, args.instance_id)
    if args.ready_file:
        # Escrita atômica: o orquestrador não lê um JSON parcialmente gravado.
        temp = args.ready_file.with_suffix(".tmp")
        salvar_json(temp, {"port": port, "instance_id": instance_id})
        temp.replace(args.ready_file)
    print(f"Servidor pronto em {args.host}:{port}", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(2).wait()


if __name__ == "__main__":
    main()
