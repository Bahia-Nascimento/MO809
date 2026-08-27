import grpc
import ml_pb2 as pb2
import ml_pb2_grpc as pb2_grpc
import time

class ExemploGRPC(object):

    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub        = pb2_grpc.MLStub(self.channel)
    
    def get_fit(self, atributos, rotulos):
        samples = []

        for data, label in zip(atributos, rotulos):
            sample = pb2.Sample(
                data=data,
                label=label
            )
            samples.append(sample)

        request = pb2.FitRequest(samples=samples)

        return self.stub.GetFit(request)
    
    def get_predict(self, atributos, rotulos):
        samples = []

        for data, label in zip(atributos, rotulos):
            sample = pb2.Sample(
                data=data,
                label=label
            )
            samples.append(sample)

        request = pb2.PredictRequest(samples=samples)

        return self.stub.GetPredict(request)
    
if __name__ == '__main__':
    client   = ExemploGRPC()
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris      = load_iris()
    atributos = iris.data
    rotulos   = iris.target
    x_treino, x_teste, y_treino, y_test = train_test_split(atributos, rotulos, test_size=0.2)

    
    while True:
        action = int(input(f'Qual ação você quer realizar?\n 1. Treinar modelo\n 2. Obter predição\n'))
        if action == 1:
            resposta = client.get_fit(x_treino, y_treino)
            print(f'Acurácia em teste: {resposta.acc}')
        elif action == 2:
            resposta = client.get_predict(x_teste, y_test)
            print(f'Rotulos: {resposta.predictions}')
            print(f'Acurácia em teste: {resposta.acc}')