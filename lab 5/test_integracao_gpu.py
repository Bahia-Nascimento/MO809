"""Teste opcional com três processos de clientes, GPU e dados sintéticos pequenos.

Não mede acurácia de CIFAR-10. Exercita a orquestração, os lotes incompletos,
o checkpoint conjunto e a execução real de M1/M2/M3 na GPU, sem download.
"""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from experimento import treinar


class IntegracaoGPU(unittest.TestCase):
    def test_tres_clientes_lotes_incompletos_checkpoint_e_teste(self):
        with tempfile.TemporaryDirectory(prefix="lab5-integracao-") as tmp:
            out = Path(tmp)
            (out / "dados").mkdir()
            rng = np.random.default_rng(42)
            for client in range(3):
                arrays = {}
                for split, n in (("train", 5), ("val", 2), ("test", 2)):
                    arrays[f"x_{split}"] = rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
                    arrays[f"y_{split}"] = rng.integers(0, 10, n)
                np.savez(out / "dados" / f"cliente_{client}.npz", **arrays)
            args = argparse.Namespace(clientes=3, epocas=1, batch_size=3,
                                      seed=42, memoria_mb=512, learning_rate=0.001)
            try:
                result = treinar(args, out, "teste-integracao-gpu")
            except Exception:
                print((out / "servidor.log").read_text())
                raise
            self.assertEqual(result["updates_m2"], 6)
            self.assertEqual(result["epocas"][0]["train_n"], 15)
            self.assertEqual(result["teste"]["n"], 6)
            self.assertTrue((out / "M2.keras").exists())
            for client in range(3):
                for name in ("M1.keras", "M3.keras", "results.csv"):
                    self.assertTrue((out / f"cliente_{client}" / name).exists())
            processos = json.loads((out / "processos.json").read_text())
            print("Processos verificados:", json.dumps(processos, indent=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
