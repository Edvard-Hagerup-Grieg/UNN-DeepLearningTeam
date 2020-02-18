from keras.layers import *
from keras.models import Model
import numpy as np


def build_conv_model_1(input_size=(28, 28, 1), number_of_classes=10, number_of_layers=1,
                        filters=[10], kernel_sizes=[(2, 2)], activation='relu'):

    input_image = Input(shape=input_size)
    x = input_image

    for layer_num in range(number_of_layers):
        x = Conv2D(filters=filters[layer_num],
                   kernel_size=kernel_sizes[layer_num],
                   activation=activation,
                   padding='same')(x)

        x = MaxPool2D(pool_size=(2, 2), strides=None, padding='same')(x)

    x = Flatten()(x)
    output = Dense(number_of_classes, activation="softmax")(x)

    model = Model(input_image, output)
    return model


def generate_model_zoo():
    models = []
    parameters_list = [[1, [32], [(3,3)], 'relu'],
                       [1, [32], [(5,5)], 'relu'],
                       [2, [32,32], [(3,3), (5,5)], 'relu'],
                       [2, [32,32], [(5,5), (3,3)], 'relu'],
                       [3, [32,32,32], [(3,3), (3,3), (3,3)], 'relu'],
                       [1, [32], [(3, 3)], 'sigmoid'],
                       [1, [32], [(5, 5)], 'sigmoid'],
                       [2, [32, 32], [(3, 3), (5, 5)], 'sigmoid'],
                       [2, [32, 32], [(5, 5), (3, 3)], 'sigmoid'],
                       [3, [32, 32, 32], [(3, 3), (3, 3), (3, 3)], 'sigmoid']]

    for parameters in parameters_list:
        models.append(build_conv_model_1(
            number_of_layers=parameters[0],
            filters=parameters[1],
            kernel_sizes=parameters[2],
            activation=parameters[3]
        ))

    return models, parameters_list

if __name__ == "__main__":
    models, parameters = generate_model_zoo()

    for model in models:
        model.summary()