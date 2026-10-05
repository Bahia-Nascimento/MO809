"""Testes matemáticos e RPCs reais, sem download e sem treino longo."""
from concurrent import futures
from pathlib import Path
import tempfile
import unittest

import grpc
import numpy as np
from config import configurar_tensorflow, GRPC_OPTIONS
tf = configurar_tensorflow(gpu=False)
from cliente import train_step, mensagem
from dados import particionar
from modelo import create_partial_model, create_server_model, create_head_model
from servidor import SplitLearningService
import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc


class Lab5Test(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.service = SplitLearningService(clientes=3, output=self.temp.name)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_OPTIONS)
        pb2_grpc.add_SplitLearningServicer_to_server(self.service, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}", options=GRPC_OPTIONS)
        self.stub = pb2_grpc.SplitLearningStub(self.channel)

    def tearDown(self):
        self.channel.close()
        self.server.stop(0).wait()
        self.temp.cleanup()

    def assert_rpc(self, code, call, request):
        with self.assertRaises(grpc.RpcError) as erro:
            call(request, timeout=5)
        self.assertEqual(erro.exception.code(), code)

    def test_split_equivale_rede_unica_em_gradientes_e_updates(self):
        m1, m3 = create_partial_model(), create_head_model()
        r1, r2, r3 = create_partial_model(), create_server_model(), create_head_model()
        for ref, real in ((r1, m1), (r2, self.service.server_model), (r3, m3)):
            ref.set_weights(real.get_weights())
        # SGD sem momentum torna a comparação dos updates proporcional à dos
        # gradientes, sem o reescalonamento do Adam esconder erros de magnitude.
        rate = 0.01
        opt1, opt3 = tf.keras.optimizers.SGD(rate), tf.keras.optimizers.SGD(rate)
        self.service.optimizer = tf.keras.optimizers.SGD(rate)
        refs = [tf.keras.optimizers.SGD(rate) for _ in range(3)]
        x = tf.random.uniform((3, 32, 32, 3))
        y = tf.constant([1, 4, 8])
        errors = []
        for step in range(2):
            with tf.GradientTape() as tape:
                a = r1(x, training=True)
                tape.watch(a)
                b = r2(a, training=True)
                tape.watch(b)
                predictions = r3(b, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, predictions))
            variables = r1.trainable_variables + r2.trainable_variables + r3.trainable_variables
            gradients = tape.gradient(loss, [a, b] + variables)
            cut_gradients, weight_gradients = gradients[:2], gradients[2:]
            # Inspeciona as duas mensagens de backward transportadas pelo gRPC.
            class RecordingStub:
                def Forward(inner, request, **kwargs):
                    return self.stub.Forward(request, **kwargs)
                def Backward(inner, request, **kwargs):
                    inner.db = np.asarray(request.values).reshape(3, 64)
                    response = self.stub.Backward(request, **kwargs)
                    inner.da = np.asarray(response.values).reshape(3, 128)
                    return response
            recording = RecordingStub()
            result = train_step(m1, m3, x, y, opt1, opt3, recording, 0, step)
            self.assertAlmostEqual(result["loss"], float(loss), places=5)
            for actual, expected in ((recording.da, cut_gradients[0]), (recording.db, cut_gradients[1])):
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                errors.append(float(np.max(np.abs(actual - expected.numpy()))))
            offset = 0
            for ref, opt in zip((r1, r2, r3), refs):
                n = len(ref.trainable_variables)
                opt.apply_gradients(zip(weight_gradients[offset:offset + n], ref.trainable_variables))
                offset += n
            for real, ref in ((m1, r1), (self.service.server_model, r2), (m3, r3)):
                for actual, expected in zip(real.get_weights(), ref.get_weights()):
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        print(f"\nErro máximo absoluto nos gradientes dos cortes: {max(errors):.3g}")
        self.assertEqual(self.service.updates, 2)

    def test_particoes_disjuntas_completas_e_reproduziveis(self):
        labels = np.repeat(np.arange(10), 103)
        treino, val = particionar(labels, 3)
        repeated = particionar(labels, 3)
        todos = np.concatenate(treino + val)
        self.assertEqual(len(todos), len(labels))
        self.assertEqual(len(np.unique(todos)), len(labels))
        for grupo, copia in zip((treino, val), repeated):
            for a, b in zip(grupo, copia):
                np.testing.assert_array_equal(a, b)
        pequenos, _ = particionar(labels, 3, limite=101)
        self.assertEqual(sum(map(len, pequenos)), 101)
        self.assertTrue(all(len(p) > 0 for p in pequenos))

    def test_protocolo_sem_rotulos(self):
        campos = {f.name for message in pb2.DESCRIPTOR.message_types_by_name.values() for f in message.fields}
        self.assertFalse(campos & {"labels", "images", "loss", "predictions"})

    def test_validacao_de_mensagens(self):
        request = mensagem(tf.ones((2, 128)), 0, 0)
        request.values[0] = float("nan")
        self.assert_rpc(grpc.StatusCode.INVALID_ARGUMENT, self.stub.Forward, request)
        request = mensagem(tf.ones((2, 127)), 0, 0)
        self.assert_rpc(grpc.StatusCode.INVALID_ARGUMENT, self.stub.Forward, request)
        request = mensagem(tf.ones((2, 128)), 9, 0)
        self.assert_rpc(grpc.StatusCode.INVALID_ARGUMENT, self.stub.Forward, request)
        self.assertEqual(self.service.updates, 0)

    def test_lote_concorrente_token_replay_e_ultimo_lote(self):
        request = mensagem(tf.ones((1, 128)), 0, 0)
        response = self.stub.Forward(request, timeout=5)
        self.assert_rpc(grpc.StatusCode.RESOURCE_EXHAUSTED, self.stub.Forward,
                        mensagem(tf.ones((2, 128)), 1, 0))
        backward = mensagem(tf.ones((1, 64)), 0, 0, "token-errado")
        self.assert_rpc(grpc.StatusCode.FAILED_PRECONDITION, self.stub.Backward, backward)
        backward.token = response.token
        returned = self.stub.Backward(backward, timeout=5)
        self.assertEqual(len(returned.values), 128)
        self.assert_rpc(grpc.StatusCode.FAILED_PRECONDITION, self.stub.Backward, backward)
        self.assert_rpc(grpc.StatusCode.FAILED_PRECONDITION, self.stub.Forward, request)
        self.assertEqual(self.service.updates, 1)
        # Um segundo cliente tem sequência própria, começando em zero.
        response = self.stub.Forward(mensagem(tf.ones((2, 128)), 1, 0), timeout=5)
        self.stub.Backward(mensagem(tf.ones((2, 64)), 1, 0, response.token), timeout=5)
        self.assertEqual(self.service.updates, 2)

    def test_timeout_abandona_tape_e_interrompe_sessao(self):
        self.stub.Forward(mensagem(tf.ones((1, 128)), 0, 0), timeout=5)
        self.service.pending["created"] -= 100
        self.assert_rpc(grpc.StatusCode.FAILED_PRECONDITION, self.stub.GetInfo, pb2.Empty())
        self.assertIsNone(self.service.pending)
        self.assertEqual(self.service.updates, 0)

    def test_avaliacao_e_checkpoint_nao_treinam(self):
        request = mensagem(tf.ones((2, 128)), 0, 0)
        before = [w.copy() for w in self.service.server_model.get_weights()]
        a = self.stub.Evaluate(request, timeout=5)
        self.stub.SaveCheckpoint(pb2.Empty(), timeout=5)
        self.service.server_model.set_weights([np.zeros_like(w) for w in before])
        self.stub.LoadCheckpoint(pb2.Empty(), timeout=5)
        b = self.stub.Evaluate(request, timeout=5)
        np.testing.assert_allclose(a.values, b.values)
        self.assert_rpc(grpc.StatusCode.FAILED_PRECONDITION, self.stub.Forward, request)
        self.assertEqual(self.service.updates, 0)
        self.assertIsNone(self.service.pending)
        for w, original in zip(self.service.server_model.get_weights(), before):
            np.testing.assert_array_equal(w, original)
        # Formato .keras usa apenas camadas padrão, sem código customizado ao abrir.
        for name, model in (("M1", create_partial_model()), ("M3", create_head_model())):
            path = Path(self.temp.name) / f"{name}.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            x = tf.ones((2,) + model.input_shape[1:])
            np.testing.assert_allclose(model(x), restored(x))


if __name__ == "__main__":
    unittest.main(verbosity=2)
