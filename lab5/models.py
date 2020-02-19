from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import *
from keras.models import Model


def load_mobilenet_model(input_size=(32, 32, 3), number_of_classes=10, weights=None):
    mobilenet_model = MobileNetV2(input_shape=input_size, weights=weights, include_top=False)
    for l in mobilenet_model.layers:
        l.trainable = False
    x = Flatten()(mobilenet_model.layers[-1].output)

    output = Dense(number_of_classes, activation="softmax")(x)

    model = Model(mobilenet_model.inputs, output)
    return model


if __name__ == "__main__":
    model = load_mobilenet_model(weights='imagenet')
    model.summary()