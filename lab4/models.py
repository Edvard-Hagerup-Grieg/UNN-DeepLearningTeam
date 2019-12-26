from keras.layers import *
from keras.models import Model
import numpy as np


def build_dense_model_1(input_size=784, number_of_classes=10, number_of_layers=1,
                        hidden_sizes=[512], activation='relu'):
    input_image = Input(shape=(input_size,))
    x = input_image

    for layer_num in range(number_of_layers):
        x = Dense(hidden_sizes[layer_num], activation=activation)(x)

    output = Dense(number_of_classes, activation="softmax")(x)

    model = Model(input_image, output)
    return model


def build_simple_autoencoder(input_size=784, number_of_layers=1,
                             hidden_sizes=[512], activation='relu'):
    input_image = Input(shape=(input_size,))
    x = input_image

    for layer_num in range(number_of_layers):
        x = Dense(hidden_sizes[layer_num], activation=activation)(x)

    for layer_num in range(number_of_layers):
        x = Dense(hidden_sizes[-1 - layer_num], activation=activation)(x)

    output = Dense(input_size, activation=activation)(x)

    model = Model(input_image, output)
    return model


def build_simple_encoder(input_size=784, number_of_layers=1,
                         hidden_sizes=[512], activation='relu'):
    input_image = Input(shape=(input_size,))
    x = input_image

    for layer_num in range(number_of_layers):
        x = Dense(hidden_sizes[layer_num], activation=activation)(x)

    encoded = x

    input_encoded = Input(shape=(hidden_sizes[-1],))
    x = input_encoded

    for layer_num in range(number_of_layers - 1):
        x = Dense(hidden_sizes[-2 - layer_num], activation=activation)(x)

    decoded = Dense(input_size, activation=activation)(x)

    encoder = Model(input_image, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_image, decoder(encoder(input_image)), name="autoencoder")

    return input_image, encoder, decoder, autoencoder


def generate_autoencoders_zoo():
    models = []
    parameters_list = [[1, [512], 'relu'],
                       [2, [512, 256], 'relu'],
                       [3, [512, 256, 64], 'relu']]

    for parameters in parameters_list:
        models.append(build_simple_encoder(
            number_of_layers=parameters[0], hidden_sizes=parameters[1], activation=parameters[2]
        ))

    return models, parameters_list


def generate_model_zoo(input_size=512):
    models = []
    parameters_list = [[1, [input_size // 2], 'relu'],
                       [2, [input_size // 2, input_size // 4], 'relu'],
                       [3, [input_size // 2, input_size // 4, input_size // 8], 'relu']]

    for parameters in parameters_list:
        models.append(
            build_dense_model_1(input_size=input_size, number_of_layers=parameters[0], hidden_sizes=parameters[1],
                                activation=parameters[2]))

    return models, parameters_list


def build_merged_models(input_image, encoder, input_size=512):
    models = []
    parameters_list = [[1, [input_size // 2], 'relu'],
                       [2, [input_size // 2, input_size // 4], 'relu'],
                       [3, [input_size // 2, input_size // 4, input_size // 8], 'relu']]

    for parameters in parameters_list:
        model = build_dense_model_1(input_size=input_size, number_of_layers=parameters[0], hidden_sizes=parameters[1],
                                    activation=parameters[2])

        new_model = Model(input_image, model(encoder(input_image)))
        models.append(new_model)
    return models, parameters_list


if __name__ == "__main__":
    models, parameters = generate_model_zoo()
