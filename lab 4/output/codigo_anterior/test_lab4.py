"""Testes do significado da loss, dados, serialização e comportamento real do RPC.

Requer os artefatos de treinar.py; use LAB4_OUTPUT para testar outra execução.
Execute: python -m unittest -v test_lab4.py
"""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import unittest
import uuid

import grpc
import numpy as np
from config import OUTPUT, configurar_tensorflow, ler_metadata
configurar_tensorflow(gpu=False)
import tensorflow as tf
from dados import criar_pares, separar_indices
from modelo import ContrastiveLoss, distancia_euclidiana
from treinar import calibrar_limiar
from mlservidor import criar_servidor
import ml_pb2
import ml_pb2_grpc


class TestDadosELoss(unittest.TestCase):
    def test_pares_e_separacao(self):
        labels = np.repeat(np.arange(10), 20)
        it, iv = separar_indices(labels)
        self.assertFalse(set(it) & set(iv))
        self.assertEqual(set(it) | set(iv), set(range(len(labels))))
        a, b, y = criar_pares(labels, 200)
        self.assertEqual(float(y.sum()), 100)
        np.testing.assert_array_equal(labels[a] == labels[b], y.ravel() == 1)
        self.assertTrue(np.all(a != b))
        for first, second in zip((a,b,y), criar_pares(labels, 200)):
            np.testing.assert_array_equal(first, second)

    def test_loss_convencao_e_gradientes(self):
        loss = ContrastiveLoss()
        # Correto: positivos próximos e negativos além da margem têm perda zero.
        self.assertAlmostEqual(float(loss(tf.constant([[1.], [0.]]), tf.constant([[0.], [1.5]]))), 0.)
        self.assertAlmostEqual(float(loss(tf.constant([[1.], [0.]]), tf.constant([[1.], [0.]]))), 1.)
        # Formas (N,) e (N,1) devem resultar no mesmo cálculo.
        self.assertAlmostEqual(float(loss(tf.constant([1.,0.]), tf.constant([[0.2],[0.3]]))), (0.04+0.49)/2, places=6)
        v = tf.Variable([[0.,0.], [1.,2.]])
        with tf.GradientTape() as tape:
            valor = tf.reduce_sum(distancia_euclidiana([v, tf.identity(v)]))
        self.assertTrue(np.isfinite(tape.gradient(valor, v).numpy()).all())

    def test_limiar_com_empates_contra_busca_exaustiva(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            d = rng.integers(0, 5, 30).astype(float)
            y = rng.integers(0, 2, 30)
            limiar, acc = calibrar_limiar(d, y)
            candidatos = np.r_[-1., np.unique(d)]
            esperado = max(np.mean((d <= t) == y) for t in candidatos)
            self.assertAlmostEqual(acc, esperado)
            self.assertAlmostEqual(np.mean((d <= limiar) == y), esperado)


class TestRPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(os.environ.get("LAB4_OUTPUT", OUTPUT))
        cls.meta = ler_metadata(cls.out)
        cls.server, port, _ = criar_servidor(cls.out, "127.0.0.1:0", timeout=1)
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        cls.stub = ml_pb2_grpc.SimilaridadeStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.server.stop(0).wait()

    def request(self, client, rodada, valor=0):
        return ml_pb2.VetorRequest(client_id=str(client), rodada=rodada,
            model_id=self.meta["model_id"], valores=[valor] * self.meta["dimensao"])

    def test_rpc_par_simetria_repeticao_e_conflito(self):
        rodada = uuid.uuid4().hex
        a, b = self.request(0, rodada, 0), self.request(1, rodada, 0.1)
        with ThreadPoolExecutor(2) as pool:
            fa = pool.submit(self.stub.EnviarVetor, a, timeout=5)
            fb = pool.submit(self.stub.EnviarVetor, b, timeout=5)
            ra, rb = fa.result(), fb.result()
        esperado = np.sqrt(self.meta["dimensao"] * 0.1**2)
        self.assertAlmostEqual(ra.distancia, esperado, places=6)
        self.assertEqual(ra, rb)
        self.assertEqual(ra.similares, esperado <= self.meta["limiar"])
        self.assertEqual(ra, self.stub.EnviarVetor(a, timeout=2))
        with self.assertRaises(grpc.RpcError) as exc:
            self.stub.EnviarVetor(self.request(0, rodada, 0.9), timeout=2)
        self.assertEqual(exc.exception.code(), grpc.StatusCode.ALREADY_EXISTS)

    def test_rejeita_vetor_e_modelo_incompativeis(self):
        for caso in ["dimensao", "nan", "modelo", "cliente"]:
            req = self.request(0, uuid.uuid4().hex)
            if caso == "dimensao": del req.valores[:]
            if caso == "nan": req.valores[0] = float("nan")
            if caso == "modelo": req.model_id = "modelo-errado"
            if caso == "cliente": req.client_id = "2"
            with self.subTest(caso=caso), self.assertRaises(grpc.RpcError) as exc:
                self.stub.EnviarVetor(req, timeout=2)
            esperado = grpc.StatusCode.FAILED_PRECONDITION if caso == "modelo" else grpc.StatusCode.INVALID_ARGUMENT
            self.assertEqual(exc.exception.code(), esperado)

    def test_cliente_ausente_expira(self):
        rodada = uuid.uuid4().hex
        with self.assertRaises(grpc.RpcError) as exc:
            self.stub.EnviarVetor(self.request(0, rodada), timeout=3)
        self.assertEqual(exc.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
        # Cliente atrasado não completa uma rodada já expirada.
        with self.assertRaises(grpc.RpcError) as exc:
            self.stub.EnviarVetor(self.request(1, rodada), timeout=2)
        self.assertEqual(exc.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)

    def test_serializacao_e_inferencia_separada(self):
        siamesa = tf.keras.models.load_model(self.out / "siamesa.keras", compile=False)
        base = tf.keras.models.load_model(self.out / "rede_base.keras", compile=False)
        comparador = tf.keras.models.load_model(self.out / "comparador.keras", compile=False)
        with np.load(self.out / "dados_clientes.npz") as data:
            a = data["imagens"][:4].astype("float32")[...,None] / 255
            b = data["imagens"][4:8].astype("float32")[...,None] / 255
        junto = siamesa([a,b], training=False).numpy()
        separado = comparador([base([a], training=False),base([b], training=False)], training=False).numpy()
        np.testing.assert_allclose(junto, separado, atol=1e-6)
        invertido = siamesa([b,a], training=False).numpy()
        np.testing.assert_allclose(junto, invertido, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
