"""Blocos M1 -> M2 -> M3 e opções de regularização locais ao cliente."""
import tensorflow as tf
from tensorflow import keras
from keras.models import Model


def create_partial_model(input_layer=None, dropout=0.0, pooling_extra=False,
                         batch_normalization=False):
    if input_layer is None:
        input_layer = keras.layers.Input(shape=(32, 32, 3))
    def convolucao(x, filters):
        if not batch_normalization:
            return keras.layers.Conv2D(filters, (3, 3), activation="relu")(x)
        x = keras.layers.Conv2D(filters, (3, 3), use_bias=False)(x)
        # As estatísticas móveis pertencem a cada cliente e são salvas com M1.
        x = keras.layers.BatchNormalization()(x)
        return keras.layers.ReLU()(x)

    layer1 = convolucao(input_layer, 32)
    layer2 = keras.layers.MaxPooling2D((2, 2))(layer1)
    layer3 = convolucao(layer2, 64)
    if pooling_extra:
        layer3 = keras.layers.MaxPooling2D((2, 2))(layer3)
    # Flatten reúne as posições espaciais em um vetor por imagem. A Dense
    # seguinte aprende a representação de 128 componentes enviada a M2.
    flattened = keras.layers.Flatten()(layer3)
    layer4 = keras.layers.Dense(128, activation="relu")(flattened)
    if dropout:
        # training=True ativa a máscara; a avaliação usa training=False.
        layer4 = keras.layers.Dropout(dropout)(layer4)
    return Model(inputs=input_layer, outputs=layer4, name="M1_cliente")


def create_server_model():
    input_layer = keras.layers.Input(shape=(128,))
    output_layer = keras.layers.Dense(64, activation="relu")(input_layer)
    return Model(input_layer, output_layer, name="M2_servidor")


def create_head_model():
    input_layer = keras.layers.Input(shape=(64,))
    output_layer = keras.layers.Dense(10, activation="softmax")(input_layer)
    return Model(input_layer, output_layer, name="M3_cliente")


def verificar_gradientes(gradientes):
    for gradiente in gradientes:
        if gradiente is None:
            raise ValueError("Grafo desconectado: gradiente None.")
        tf.debugging.assert_all_finite(gradiente, "Gradiente não finito.")
