from dataset import load_dataset
from models import load_mobilenet_model
from sklearn.model_selection import train_test_split
import keras.callbacks as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def save_history_img(history, model_name='model'):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("pictures\\" + model_name)
    plt.clf()

def train_mobilenet_model(x_train, x_val, x_test, y_train, y_val, y_test, weights=None, trainable=True, model_name="model"):
    model = load_mobilenet_model(input_size=x_train.shape[1:], weights=weights, trainable=trainable)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])

    best_weights_filepath = './best_weights.hdf5'

    earlyStopping = K.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    mcp_save = K.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    history = model.fit(x_train, y_train, batch_size=2000, validation_data=(x_val, y_val), epochs=150, callbacks=[earlyStopping, mcp_save], verbose=1)

    save_history_img(history, model_name)

    y_val_cat_prob = model.predict(x_test)
    y_prediction = np.argmax(y_val_cat_prob, axis=1)
    y_labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_labels, y_prediction)
    return accuracy


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    test_accuracy = train_mobilenet_model(x_train, x_val, x_test, y_train, y_val, y_test, weights='imagenet', trainable=False)
    print("Test accuracy: ", test_accuracy)