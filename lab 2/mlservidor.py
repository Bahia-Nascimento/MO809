import threading
from concurrent import futures

import grpc
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import ml_pb2 as pb2
import ml_pb2_grpc as pb2_grpc


NUM_CLIENTS = 5
MODEL = LogisticRegression(max_iter=1000, random_state=42)

DIGITS = load_digits()
X_TREINO, X_TESTE, Y_TREINO, Y_TESTE = train_test_split(
    DIGITS.data,
    DIGITS.target % 2,
    test_size=0.2,
    random_state=42,
    stratify=DIGITS.target % 2,
)


class ExemploServer(pb2_grpc.MLServicer):
    def __init__(self):
        self.condition = threading.Condition()
        self.updates = {}
        self.results = {}

    def GetTrainingData(self, request, context):
        client_id = request.client_id
        if not 0 <= client_id < NUM_CLIENTS:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'client_id inválido')

        samples = [
            pb2.Sample(data=data, label=label)
            for index, (data, label) in enumerate(zip(X_TREINO, Y_TREINO))
            if index % NUM_CLIENTS == client_id
        ]
        return pb2.TrainingDataResponse(samples=samples)

    def SubmitModelUpdate(self, request, context):
        client_id = request.client_id
        if not 0 <= client_id < NUM_CLIENTS:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'client_id inválido')

        update = (
            np.asarray(request.weights, dtype=np.float64),
            np.asarray(request.intercept, dtype=np.float64),
        )

        with self.condition:
            if client_id in self.updates:
                context.abort(grpc.StatusCode.ALREADY_EXISTS, 'cliente já enviou atualização')

            self.updates[client_id] = update
            if len(self.updates) == NUM_CLIENTS:
                self._process_updates()
                self.condition.notify_all()
            else:
                self.condition.wait_for(lambda: len(self.results) == NUM_CLIENTS)

            accepted, accuracy, accuracy_without_detection = self.results[client_id]
            return pb2.ModelUpdateResponse(
                accepted=accepted,
                global_acc=accuracy,
                global_acc_without_detection=accuracy_without_detection,
            )

    def _process_updates(self):
        client_ids = sorted(self.updates)
        weights = np.stack([self.updates[client_id][0] for client_id in client_ids])
        intercepts = np.stack([self.updates[client_id][1] for client_id in client_ids])

        MODEL.fit(X_TREINO, Y_TREINO)
        client_accuracies = []
        for client_weights, client_intercept in zip(weights, intercepts):
            client_model = LogisticRegression(max_iter=1000, random_state=42)
            client_model.fit(X_TREINO, Y_TREINO)
            client_model.coef_ = client_weights.reshape(client_model.coef_.shape)
            client_model.intercept_ = client_intercept
            client_accuracies.append(client_model.score(X_TESTE, Y_TESTE))

        random_accuracy = 1 / len(np.unique(Y_TESTE))
        accepted_mask = np.asarray(client_accuracies) > random_accuracy

        if not np.any(accepted_mask):
            raise RuntimeError('nenhum modelo superou a acurácia de um chute aleatório')

        all_weights = np.mean(weights, axis=0).reshape(MODEL.coef_.shape)
        all_intercept = np.mean(intercepts, axis=0)
        accuracy_without_detection = self._score_parameters(all_weights, all_intercept)

        MODEL.coef_ = np.mean(weights[accepted_mask], axis=0).reshape(MODEL.coef_.shape)
        MODEL.intercept_ = np.mean(intercepts[accepted_mask], axis=0)
        accuracy = float(MODEL.score(X_TESTE, Y_TESTE))

        self.results = {
            client_id: (bool(accepted), accuracy, accuracy_without_detection)
            for client_id, accepted in zip(client_ids, accepted_mask)
        }

    def _score_parameters(self, weights, intercept):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_TREINO, Y_TREINO)
        model.coef_ = weights
        model.intercept_ = intercept
        return float(model.score(X_TESTE, Y_TESTE))

    def GetFit(self, request, context):
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            'O treinamento agora acontece localmente nos clientes',
        )

    def GetPredict(self, request, context):
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            'A agregação federada ainda não disponibiliza predição',
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=NUM_CLIENTS))
    pb2_grpc.add_MLServicer_to_server(ExemploServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print(f'Server started at 50051; aguardando {NUM_CLIENTS} clientes')
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
