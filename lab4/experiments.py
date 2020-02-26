import keras.callbacks as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset import load_dataset_vec, load_dataset_img
from models import generate_model_dense_zoo_unsupervised, generate_model_conv_zoo_unsupervised


def save_history_img(history, number, name='model'):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("pictures\\" + name + "_" + str(number))
    plt.clf()


def save_history_img_enc(history, number, name='model'):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("pictures\\" + name + "_" + str(number))
    plt.clf()


def train_models(models, x, y, name='model'):
    best_weights_filepath = './best_weights.hdf5'
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42)

    for i, model in enumerate(models):
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])

        earlyStopping = K.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        mcp_save = K.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='auto')
        history = model.fit(x, y, batch_size=2000, validation_data=(x_val, y_val), epochs=150,
                            callbacks=[earlyStopping, mcp_save], verbose=2)

        save_history_img(history, i, name)

        model.load_weights(best_weights_filepath)


def test_models(models, x, y, name='model'):
    accuracy_list = []
    for i, model in enumerate(models):
        y_val_cat_prob = model.predict(x)
        y_prediction = np.argmax(y_val_cat_prob, axis=1)
        y_labels = np.argmax(y, axis=1)

        accuracy_list.append(accuracy_score(y_labels, y_prediction))
    accuracy_list = np.array(accuracy_list)

    np.savetxt("pictures\\" + name + "accuracy.csv", accuracy_list, delimiter=";", fmt='%f')


def train_unsupervised_models(models, x, y):
    best_weights_filepath = './best_weights.hdf5'
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42)

    for i, auroencoder in enumerate(models):
        model = auroencoder
        model.compile(loss='mean_squared_error', optimizer='SGD', metrics=["mean_squared_error"])

        earlyStopping = K.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        mcp_save = K.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='auto')
        history = model.fit(x, y, batch_size=2000, validation_data=(x_val, y_val), epochs=50,
                            callbacks=[earlyStopping, mcp_save], verbose=1)

        save_history_img_enc(history, i)

        model.load_weights(best_weights_filepath)


if __name__ == "__main__":
    autoencoders, models, params = generate_model_conv_zoo_unsupervised()
    (x_train, y_train), (x_test, y_test) = load_dataset_img()
    train_unsupervised_models(autoencoders, x_train, x_train)

    train_models(models, x_train, y_train, "new_moldel")
    test_models(models, x_test, y_test, "new_model")
