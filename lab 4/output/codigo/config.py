"""Configuração comum. Variáveis de threads devem existir ANTES de importar TensorFlow."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
for name in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(name, "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configurar_tensorflow(gpu=False, seed=42):
    # Inferência de uma imagem é pequena: clientes/servidor usam CPU por padrão.
    # O treinamento usa GPU; não deixamos vários processos disputarem sua memória.
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    return tf


def hash_modelo(path):
    # Vetores de modelos distintos não são necessariamente comparáveis.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ler_metadata(output=OUTPUT):
    return json.loads((Path(output) / "metadata.json").read_text(encoding="utf-8"))


def salvar_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
