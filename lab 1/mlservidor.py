import grpc
from concurrent import futures
import time

import ml_pb2 as pb2
import ml_pb2_grpc as pb2_grpc

from sklearn.tree import DecisionTreeClassifier
MODEL = DecisionTreeClassifier(random_state=42)

class ExemploServer(pb2_grpc.MLServicer):

    def GetFit(self, request, context):
        input_data = []
        input_labels = []

        for sample in request.samples:
            input_data.append(list(sample.data))
            input_labels.append(sample.label)

        MODEL.fit(input_data, input_labels)

        accuracy = MODEL.score(input_data, input_labels)

        return pb2.FitResponse(acc=accuracy)
    
    def GetPredict(self, request, context):
        input_data = []
        input_labels = []

        for sample in request.samples:
            input_data.append(list(sample.data))
            input_labels.append(sample.label)

        predictions = MODEL.predict(input_data)

        accuracy = MODEL.score(input_data, input_labels)

        return pb2.PredictResponse(
            predictions=predictions.tolist(),
            acc=accuracy
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MLServicer_to_server(ExemploServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()