"""Fashion MNIST e pares: similar significa pertencer à MESMA CLASSE de roupa."""
import numpy as np

CLASSES = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco", "Sandália",
           "Camisa", "Tênis", "Bolsa", "Bota"]


def separar_indices(labels, seed=42):
    """Reserva 20% de cada classe ANTES de criar pares, evitando vazamento."""
    rng = np.random.default_rng(seed)
    treino, validacao = [], []
    for classe in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == classe))
        n = len(indices) // 5
        validacao.extend(indices[:n])
        treino.extend(indices[n:])
    return rng.permutation(treino), rng.permutation(validacao)


def criar_pares(labels, quantidade, seed=42):
    """Retorna índices A/B e rótulos (N,1), sem duplicar as imagens na memória.

    Metade dos pares tem mesma classe. Pares positivos nunca repetem a própria
    imagem; os negativos usam outra classe. Repetições de pares são possíveis.
    """
    if quantidade < 2 or quantidade % 2:
        raise ValueError("A quantidade de pares deve ser par e >= 2.")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    grupos = {c: np.flatnonzero(labels == c) for c in classes}
    if len(classes) < 2 or any(len(g) < 2 for g in grupos.values()):
        raise ValueError("São necessárias duas classes com pelo menos duas imagens cada.")
    a = rng.integers(len(labels), size=quantidade)
    y = np.tile([1, 0], quantidade // 2).astype("float32")
    b = np.empty(quantidade, dtype="int32")
    for i, anchor in enumerate(a):
        classe = labels[anchor]
        if y[i] == 1:
            candidato = rng.choice(grupos[classe])
            while candidato == anchor:
                candidato = rng.choice(grupos[classe])
        else:
            outra = rng.choice(classes[classes != classe])
            candidato = rng.choice(grupos[outra])
        b[i] = candidato
    return a.astype("int32"), b, y[:, None]


def dataset_pares(imagens, pares, batch_size, treino=False, seed=42):
    """tf.data monta apenas o lote atual; pixels uint8 viram float32 em [0,1]."""
    import tensorflow as tf
    a, b, labels = pares
    banco = tf.convert_to_tensor(imagens)
    ds = tf.data.Dataset.from_tensor_slices((a, b, labels))
    if treino:
        ds = ds.shuffle(len(a), seed=seed, reshuffle_each_iteration=True)
    def preparar(i, j, y):
        x1 = tf.cast(tf.gather(banco, i), tf.float32)[..., None] / 255.0
        x2 = tf.cast(tf.gather(banco, j), tf.float32)[..., None] / 255.0
        return (x1, x2), y
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.map(preparar, num_parallel_calls=2).batch(batch_size).prefetch(1).with_options(options)
