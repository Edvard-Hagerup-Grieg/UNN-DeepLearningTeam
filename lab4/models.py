from keras.layers import *
from keras.models import Model
import numpy as np


def build_simple_autoencoder(input_size=784, number_of_layers=1,
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


def generate_model_zoo_unsupervised():
    models = []
    autoencoders = []
    parameters_list = [[2, [512, 256], 'relu'],
                       [3, [512, 256, 64], 'linear'],
                       [2, [512, 256], 'linear'],
                       [3, [512, 256, 64], 'relu']]

    for parameters in parameters_list:
        input_image, encoder, decoder, autoencoder = build_simple_autoencoder(number_of_layers=parameters[0],
                                                                              hidden_sizes=parameters[1],
                                                                              activation=parameters[2])

        model_input = Input(shape=(parameters[1][-1],))
        x = Dense(10, activation="softmax")(model_input)
        model = Model(model_input, x)
        classifier = Model(input_image, model(encoder(input_image)))
        models.append(classifier)
        autoencoders.append(autoencoder)

    return autoencoders, models, parameters_list


if __name__ == "__main__":
    models, parameters = generate_model_zoo_unsupervised()
