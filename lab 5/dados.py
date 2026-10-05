"""CIFAR-10: partições locais disjuntas, estratificadas e reproduzíveis."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from config import configurar_tensorflow, salvar_json


def selecionar(indices, labels, limite, rng):
    """Subamostra balanceada; limite=0 conserva todos os exemplos."""
    if not limite or limite >= len(indices):
        return rng.permutation(indices)
    classes = [rng.permutation(indices[labels[indices] == c]) for c in range(10)]
    # Round-robin entre classes, sem reposição, incluindo tamanhos não múltiplos de 10.
    ordem = [grupo[i] for i in range(max(map(len, classes)))
             for grupo in classes if i < len(grupo)]
    return rng.permutation(np.asarray(ordem[:limite]))


def particionar(labels, clientes, seed=42, validacao=True, limite=0):
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    treino, val = [], []
    for classe in range(10):
        indices = rng.permutation(np.flatnonzero(labels == classe))
        n_val = len(indices) // 10 if validacao else 0
        val.extend(indices[:n_val])
        treino.extend(indices[n_val:])
    treino = selecionar(np.asarray(treino, dtype=int), labels, limite, rng)
    conjuntos = []
    for indices in (treino, np.asarray(val, dtype=int)):
        locais = [[] for _ in range(clientes)]
        for classe in range(10):
            for i, parte in enumerate(np.array_split(indices[labels[indices] == classe], clientes)):
                locais[i].extend(parte)
        conjuntos.append([rng.permutation(np.asarray(p, dtype=int)) for p in locais])
    return conjuntos


def preparar(output, clientes=2, seed=42, limite_treino=0, limite_teste=0):
    # Este processo só baixa/prepara arquivos e termina antes do treino GPU.
    tf = configurar_tensorflow(gpu=False, seed=seed)
    (x, y), (xt, yt) = tf.keras.datasets.cifar10.load_data()
    y, yt = y.reshape(-1), yt.reshape(-1)
    treino, val = particionar(y, clientes, seed, limite=limite_treino)
    teste, _ = particionar(yt, clientes, seed + 1, validacao=False, limite=limite_teste)
    output = Path(output)
    pasta = output / "dados"
    pasta.mkdir(parents=True, exist_ok=False)
    manifesto = {"dataset": "CIFAR-10", "seed": seed, "limite_treino": limite_treino,
                 "limite_teste": limite_teste, "clientes": []}
    for i in range(clientes):
        if not len(treino[i]) or not len(teste[i]):
            raise ValueError("Limites insuficientes para dar exemplos a todos os clientes.")
        arquivo = pasta / f"cliente_{i}.npz"
        np.savez(arquivo, x_train=x[treino[i]], y_train=y[treino[i]],
                 x_val=x[val[i]], y_val=y[val[i]], x_test=xt[teste[i]], y_test=yt[teste[i]],
                 train_indices=treino[i], val_indices=val[i], test_indices=teste[i])
        manifesto["clientes"].append({"id": i, "treino": len(treino[i]),
            "validacao": len(val[i]), "teste": len(teste[i]),
            "classes_treino": np.bincount(y[treino[i]], minlength=10).tolist(),
            "sha256": hashlib.sha256(arquivo.read_bytes()).hexdigest()})
    salvar_json(output / "particoes.json", manifesto)
    print(f"Dados preparados: {[len(p) for p in treino]} exemplos de treino por cliente.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clientes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limite-treino", type=int, default=0)
    parser.add_argument("--limite-teste", type=int, default=0)
    args = parser.parse_args()
    if args.clientes < 1 or min(args.limite_treino, args.limite_teste) < 0:
        parser.error("Clientes >= 1; limites >= 0.")
    preparar(**vars(args))
