from lab3.models import generate_model_zoo
from lab3.dataset import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import keras.callbacks as K


def save_history_img(history, number):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("pictures\\model_" + str(number))
    plt.clf()


def calculate_accuracy(model, x, y):
    y_prediction = np.argmax(model.predict(x), axis=1)
    y_labels = np.argmax(y, axis=1)

    val_accuracy = accuracy_score(y_labels, y_prediction)

    return val_accuracy


def train_models(models, x, y):
    best_weights_filepath = './best_weights.hdf5'
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42)

    for i, model in enumerate(models):
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])

        earlyStopping = K.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        mcp_save = K.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        history = model.fit(x, y, batch_size=2000, validation_data=(x_val, y_val), epochs=150, callbacks=[earlyStopping, mcp_save], verbose=2)

        save_history_img(history, i)

        model.load_weights(best_weights_filepath)


def test_models(models, x, y):
    accuracy_list = []
    for i, model in enumerate(models):
        y_val_cat_prob = model.predict(x)
        y_prediction = np.argmax(y_val_cat_prob, axis=1)
        y_labels = np.argmax(y, axis=1)

        accuracy_list.append(accuracy_score(y_labels, y_prediction))
    accuracy_list = np.array(accuracy_list)

    np.savetxt("pictures\\accuracy.csv", accuracy_list, delimiter=";", fmt='%f')


if __name__ == "__main__":
    models, parameters = generate_model_zoo()
    (x_train, y_train), (x_test, y_test) = load_dataset()

    train_models(models, x_train, y_train)
    test_models(models, x_test, y_test)