"""Configuração antes de importar TensorFlow, também nos processos Ray."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
MAX_BATCH = 256
GRPC_OPTIONS = [("grpc.max_send_message_length", 4 * 1024 * 1024),
                ("grpc.max_receive_message_length", 4 * 1024 * 1024),
                ("grpc.enable_retries", 0)]
for key in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(key, "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configurar_tensorflow(gpu=True, seed=42, memoria_mb=1024):
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    dispositivos = tf.config.list_physical_devices("GPU")
    if gpu and not dispositivos:
        raise RuntimeError("GPU indisponível. Execute com o venv-gpu no WSL.")
    if gpu:
        # Ray reserva frações LÓGICAS; não limita a VRAM. Cada processo TensorFlow
        # recebe seu próprio teto antes de criar tensores ou inicializar CUDA.
        tf.config.set_logical_device_configuration(dispositivos[0], [
            tf.config.LogicalDeviceConfiguration(memory_limit=memoria_mb)])
    tf.keras.utils.set_random_seed(seed)
    return tf


def salvar_json(path, objeto):
    Path(path).write_text(json.dumps(objeto, indent=2, ensure_ascii=False), encoding="utf-8")
