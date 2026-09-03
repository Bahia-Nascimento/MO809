import threading
from concurrent import futures

import grpc
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import ml_pb2 as pb2
import ml_pb2_grpc as pb2_grpc


NUM_CLIENTS = 5
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
        self.client_data = {}
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

    def SubmitDataUpdate(self, request, context):
        client_id = request.client_id
        if not 0 <= client_id < NUM_CLIENTS:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'client_id inválido')

        data = np.asarray([sample.data for sample in request.samples], dtype=np.float64)
        labels = np.asarray([sample.label for sample in request.samples], dtype=np.int64)

        with self.condition:
            if client_id in self.client_data:
                context.abort(grpc.StatusCode.ALREADY_EXISTS, 'cliente já enviou dados')

            self.client_data[client_id] = (data, labels)
            if len(self.client_data) == NUM_CLIENTS:
                self._process_updates()
                self.condition.notify_all()
            else:
                self.condition.wait_for(lambda: len(self.results) == NUM_CLIENTS)

            accepted, client_accuracy, accuracy, accuracy_without_detection = self.results[client_id]
            return pb2.DataUpdateResponse(
                accepted=accepted,
                global_acc=accuracy,
                global_acc_without_detection=accuracy_without_detection,
                client_acc=client_accuracy,
            )

    def _new_model(self):
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    def _process_updates(self):
        client_ids = sorted(self.client_data)
        client_accuracies = []
        for client_id in client_ids:
            data, labels = self.client_data[client_id]
            client_model = self._new_model()
            client_model.fit(data, labels)
            client_accuracies.append(client_model.score(X_TESTE, Y_TESTE))

        random_accuracy = 1 / len(np.unique(Y_TESTE))
        accepted_mask = np.asarray(client_accuracies) > random_accuracy
        if not np.any(accepted_mask):
            raise RuntimeError('nenhum cliente superou a acurácia de um chute aleatório')

        all_data = np.concatenate([self.client_data[client_id][0] for client_id in client_ids])
        all_labels = np.concatenate([self.client_data[client_id][1] for client_id in client_ids])
        model_without_detection = self._new_model()
        model_without_detection.fit(all_data, all_labels)
        accuracy_without_detection = float(model_without_detection.score(X_TESTE, Y_TESTE))

        accepted_ids = [client_id for client_id, accepted in zip(client_ids, accepted_mask) if accepted]
        accepted_data = np.concatenate([self.client_data[client_id][0] for client_id in accepted_ids])
        accepted_labels = np.concatenate([self.client_data[client_id][1] for client_id in accepted_ids])
        model_with_detection = self._new_model()
        model_with_detection.fit(accepted_data, accepted_labels)
        accuracy = float(model_with_detection.score(X_TESTE, Y_TESTE))

        self.results = {
            client_id: (
                bool(accepted),
                float(client_accuracy),
                accuracy,
                accuracy_without_detection,
            )
            for client_id, accepted, client_accuracy in zip(
                client_ids,
                accepted_mask,
                client_accuracies,
            )
        }

    def GetFit(self, request, context):
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            'Use SubmitDataUpdate: o treinamento é realizado no servidor',
        )

    def GetPredict(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, 'Predição ainda não implementada')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=NUM_CLIENTS))
    pb2_grpc.add_MLServicer_to_server(ExemploServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print(f'Server started at 50051; aguardando {NUM_CLIENTS} clientes')
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
