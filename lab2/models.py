from keras.layers import *
from keras.models import Model
import numpy as np


def build_dense_model_1(input_size=784, number_of_classes=10, number_of_layers=1,
                        hidden_sizes=[512], activation='relu'):

    input_image = Input(shape=(input_size,))
    x = input_image

    for layer_num in range(number_of_layers):
        x = Dense(hidden_sizes[layer_num], activation=activation)(x)

    output = Dense(number_of_classes, activation="sigmoid")(x)

    model = Model(input_image, output)
    return model


def generate_model_zoo():
    models = []
    parameters_list = [[1, [256], 'relu'],
              [1, [512], 'relu'],
              [2, [512, 256], 'relu'],
              [3, [512, 256, 64], 'relu'],
              [1, [256], 'linear'],
              [1, [512], 'linear'],
              [2, [512, 256], 'linear'],
              [3, [512, 256, 64], 'linear'],
              [1, [256], 'sigmoid'],
              [1, [512], 'sigmoid'],
              [2, [512, 256], 'sigmoid'],
              [3, [512, 256, 64], 'sigmoid']]

    for parameters in parameters_list:
        models.append(build_dense_model_1(
            number_of_layers=parameters[0], hidden_sizes=parameters[1], activation=parameters[2]
        ))

    return models, parameters_list

if __name__ == "__main__":
    models, parameters = generate_model_zoo()