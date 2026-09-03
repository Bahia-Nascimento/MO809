import sys

import grpc
import numpy as np
from sklearn.linear_model import LogisticRegression

import ml_pb2 as pb2
import ml_pb2_grpc as pb2_grpc


NUM_CLIENTS = 5
FEATURE_MIN = 0.0
FEATURE_MAX = 16.0

BEHAVIOR_NAMES = {
    0: 'normal',
    1: 'ruído em todas as features',
    2: 'ruído em uma feature',
    3: 'rótulos invertidos',
}


class Cliente:
    def __init__(self, client_id):
        self.client_id = client_id
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = pb2_grpc.MLStub(self.channel)

    def get_training_data(self):
        request = pb2.TrainingDataRequest(client_id=self.client_id)
        return self.stub.GetTrainingData(request)

    def submit_model_update(self, model):
        request = pb2.ModelUpdateRequest(
            client_id=self.client_id,
            weights=model.coef_.ravel().tolist(),
            intercept=model.intercept_.tolist(),
        )
        return self.stub.SubmitModelUpdate(request)


def executar_cliente(client_id, behavior_id):
    if not 0 <= client_id < NUM_CLIENTS:
        raise ValueError(f'client_id deve estar entre 0 e {NUM_CLIENTS - 1}')
    if behavior_id not in BEHAVIOR_NAMES:
        raise ValueError('behavior_id deve estar entre 0 e 3')

    client = Cliente(client_id)
    resposta = client.get_training_data()
    atributos = np.array([sample.data for sample in resposta.samples])
    rotulos = np.array([sample.label for sample in resposta.samples])

    rng = np.random.default_rng(1000 + client_id)
    if behavior_id == 1:
        atributos = rng.uniform(
            FEATURE_MIN,
            FEATURE_MAX,
            size=atributos.shape,
        )
    elif behavior_id == 2:
        feature_id = int(rng.integers(atributos.shape[1]))
        atributos[:, feature_id] = rng.uniform(
            FEATURE_MIN,
            FEATURE_MAX,
            size=len(atributos),
        )
    elif behavior_id == 3:
        rotulos = 1 - rotulos

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(atributos, rotulos)
    resposta_agregacao = client.submit_model_update(model)

    return {
        'client_id': client_id,
        'behavior_id': behavior_id,
        'samples': len(resposta.samples),
        'accepted': resposta_agregacao.accepted,
        'acc_with_detection': resposta_agregacao.global_acc,
        'acc_without_detection': resposta_agregacao.global_acc_without_detection,
    }


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(
            f'uso: python mlcliente.py <client_id: 0-{NUM_CLIENTS - 1}> '
            '<behavior_id: 0-3>'
        )

    result = executar_cliente(int(sys.argv[1]), int(sys.argv[2]))
    print(
        f"Cliente {result['client_id']} ({BEHAVIOR_NAMES[result['behavior_id']]}): "
        f"{result['samples']} amostras locais; "
        f"update aceito={result['accepted']}; "
        f"acurácia com detecção={result['acc_with_detection']:.4f}; "
        f"acurácia sem detecção={result['acc_without_detection']:.4f}"
    )
