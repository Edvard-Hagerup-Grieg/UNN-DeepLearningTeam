from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


def load_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train_28 = x_train.reshape(60000, 28, 28, 1)
    x_test_28 = x_test.reshape(10000, 28, 28, 1)
    x_train_28 = x_train_28.astype('float32')
    x_test_28 = x_test_28.astype('float32')
    x_train_28 /= 255.0
    x_test_28 /= 255.0

    x_train_32 = np.zeros((x_train_28.shape[0], x_train_28.shape[1] + 4, x_train_28.shape[2] + 4, 3))
    x_test_32 = np.zeros((x_test_28.shape[0], x_test_28.shape[1] + 4, x_test_28.shape[2] + 4, 3))

    for i in range(3):
        x_train_32[:, 2:-2, 2:-2, i:i+1] = x_train_28
        x_test_32[:, 2:-2, 2:-2, i:i+1] = x_test_28


    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train_32, y_train), (x_test_32, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    plt.imshow(x_train[0,:,:,0], cmap="Greys")
    plt.show()