from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def load_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    plt.imshow(x_train[0,:,:,0], cmap="Greys")
    plt.show()