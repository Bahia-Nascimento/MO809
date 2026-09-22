"""Functional API: uma rede base é chamada duas vezes, compartilhando os pesos."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Lab4")
def distancia_euclidiana(vetores):
    a, b = vetores
    soma = tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True)
    # A derivada de sqrt(0) é problemática. O piso evita NaN no treinamento.
    # Por isso vetores iguais retornam sqrt(epsilon), aproximadamente 0.000316.
    return tf.sqrt(tf.maximum(soma, tf.keras.backend.epsilon()))


@tf.keras.utils.register_keras_serializable(package="Lab4")
class ContrastiveLoss(tf.keras.losses.Loss):
    """y=1 aproxima; y=0 afasta até a margem. Corrige a inversão do PDF."""
    def __init__(self, margem=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margem = margem

    def call(self, y_true, y_pred):
        # Ambas as formas precisam ser (lote,1), evitando broadcasting (lote,lote).
        y = tf.reshape(tf.cast(y_true, y_pred.dtype), tf.shape(y_pred))
        perdas = y * tf.square(y_pred) + (1 - y) * tf.square(tf.maximum(self.margem - y_pred, 0))
        return tf.reduce_mean(perdas, axis=-1)

    def get_config(self):
        return {**super().get_config(), "margem": self.margem}


def criar_comparador(dimensao):
    a = tf.keras.Input((dimensao,), name="vetor_a")
    b = tf.keras.Input((dimensao,), name="vetor_b")
    # Função nomeada e registrada: podemos salvar/recarregar sem safe_mode=False.
    distancia = tf.keras.layers.Lambda(distancia_euclidiana, output_shape=(1,), name="distancia")([a, b])
    return tf.keras.Model([a, b], distancia, name="comparador")


def criar_modelos(hp):
    # O Tuner escolhe valores neste espaço; não são os parâmetros finais fixados.
    filtros_1 = hp.Choice("filtros_1", [16, 32])
    filtros_2 = hp.Choice("filtros_2", [32, 64])
    unidades = hp.Choice("unidades_densas", [64, 128])
    dimensao = hp.Choice("dimensao_embedding", [16, 32, 64])
    dropout = hp.Choice("dropout", [0.0, 0.1, 0.2, 0.3, 0.4])
    usar_bn = hp.Boolean("batch_normalization")
    entrada = tf.keras.Input((28, 28, 1), name="imagem")
    # BN normaliza ativações usando o lote no treino e médias móveis na inferência.
    # Fica ANTES da ReLU. Quando ativa, seu deslocamento beta substitui o bias.
    # Essas estatísticas são compartilhadas pelos dois ramos e salvas com o modelo.
    for i, filtros in enumerate([filtros_1, filtros_2], 1):
        x_entrada = entrada if i == 1 else x
        x = tf.keras.layers.Conv2D(filtros, 3, padding="same", use_bias=not usar_bn,
                                    name=f"conv_{i}")(x_entrada)
        if usar_bn:
            x = tf.keras.layers.BatchNormalization(name=f"bn_conv_{i}")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(unidades, use_bias=not usar_bn)(x)
    if usar_bn:
        x = tf.keras.layers.BatchNormalization(name="bn_densa")(x)
    x = tf.keras.layers.Activation("relu")(x)
    # Dropout desliga unidades aleatoriamente no TREINO; na inferência fica inativo.
    x = tf.keras.layers.Dropout(dropout)(x)
    vetor = tf.keras.layers.Dense(dimensao, name="embedding")(x)
    base = tf.keras.Model(entrada, vetor, name="rede_base")
    a = tf.keras.Input((28, 28, 1), name="imagem_a")
    b = tf.keras.Input((28, 28, 1), name="imagem_b")
    comparador = criar_comparador(dimensao)
    # É o MESMO objeto base: criar duas redes treinaria pesos independentes.
    siamesa = tf.keras.Model([a, b], comparador([base(a), base(b)]), name="siamesa")
    return base, comparador, siamesa


def construir_para_tuner(hp):
    _, _, modelo = criar_modelos(hp)
    taxa = hp.Choice("learning_rate", [0.001, 0.0003])
    # Margem fixa em 1: mudar sua escala entre trials tornaria val_loss incomparável.
    modelo.compile(optimizer=tf.keras.optimizers.Adam(taxa),
                   loss=ContrastiveLoss(margem=1.0), jit_compile=False)
    return modelo
