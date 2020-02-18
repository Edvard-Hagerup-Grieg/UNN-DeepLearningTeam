from keras.layers import *
from keras.models import Model


def build_dense_autoencoder(input_size=784, number_of_layers=1,
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


def build_conv_autoencoder(input_size=(28, 28, 1), number_of_layers=1,
                           filters=[10], kernel_sizes=[(2, 2)], activation='relu'):

    input_image = Input(shape=input_size)
    x = input_image

    for layer_num in range(number_of_layers):
        x = Conv2D(filters=filters[layer_num],
                   kernel_size=kernel_sizes[layer_num],
                   activation=activation,
                   padding='same')(x)

        x = MaxPool2D(pool_size=(2, 2), strides=None, padding='same')(x)

    encoded = x

    bottle_neck_size = 28//(2**number_of_layers)
    input_encoded = Input(shape=(bottle_neck_size, bottle_neck_size, filters[-1]))
    x = input_encoded

    for layer_num in range(number_of_layers - 1):
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(filters=filters[-layer_num],
                   kernel_size=kernel_sizes[-layer_num],
                   activation=activation,
                   padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(filters= 1,
               kernel_size=kernel_sizes[-1],
               activation=activation,
               padding='same')(x)

    decoded = x

    encoder = Model(input_image, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_image, decoder(encoder(input_image)), name="autoencoder")

    return input_image, encoder, decoder, autoencoder


def generate_model_dense_zoo_unsupervised():
    models = []
    autoencoders = []
    parameters_list = [[2, [512, 256], 'relu'],
                       [3, [512, 256, 64], 'linear'],
                       [2, [512, 256], 'linear'],
                       [3, [512, 256, 64], 'relu']]

    for parameters in parameters_list:
        input_image, encoder, decoder, autoencoder = build_dense_autoencoder(number_of_layers=parameters[0],
                                                                              hidden_sizes=parameters[1],
                                                                              activation=parameters[2])

        model_input = Input(shape=(parameters[1][-1],))
        x = Dense(10, activation="softmax")(model_input)
        model = Model(model_input, x)
        classifier = Model(input_image, model(encoder(input_image)))

        models.append(classifier)
        autoencoders.append(autoencoder)

    return autoencoders, models, parameters_list


def generate_model_conv_zoo_unsupervised():
    models = []
    autoencoders = []
    parameters_list = [[1, [32], [(3,3)], 'relu'],
                       [1, [32], [(5,5)], 'relu'],
                       [2, [32,32], [(3,3), (5,5)], 'relu'],
                       [1, [32], [(3, 3)], 'sigmoid']]

    for parameters in parameters_list:
        input_image, encoder, decoder, autoencoder = build_conv_autoencoder(number_of_layers=parameters[0],
                                                                            filters=parameters[1],
                                                                            kernel_sizes=parameters[2],
                                                                            activation=parameters[3])

        bottle_neck_size = 28 // (2 ** parameters[0])
        model_input = Input(shape=(bottle_neck_size, bottle_neck_size, parameters[1][-1]))
        x = Flatten()(model_input)
        model_output = Dense(10, activation="softmax")(x)
        model = Model(model_input, model_output)
        classifier = Model(input_image, model(encoder(input_image)))

        models.append(classifier)
        autoencoders.append(autoencoder)

    return autoencoders, models, parameters_list


if __name__ == "__main__":
    autoencoders_dense, models_dense, parameters_dense = generate_model_dense_zoo_unsupervised()
    print(parameters_dense)

    autoencoders_conv, models_conv, parameters_conv = generate_model_conv_zoo_unsupervised()
    print(parameters_conv)
