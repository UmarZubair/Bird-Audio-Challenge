from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import utils
import keras
import config
from keras.preprocessing.image import ImageDataGenerator


BEST_MODEL_FILE = "./models/best_cnn_model_trained.hdf5"
EPOCHS = 50

X, Y = utils.load_training_data()
X_test = utils.load_testing_data()
if config.options()["PSEUDO_LABELLING"]:
    x_pseudo_label, y_pseudo_label = utils.load_pseudo_test_data()
    X = np.concatenate(X, x_pseudo_label)
    Y = np.concatenate(Y, y_pseudo_label)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


def get_cnn_model(in_data, out_data):
    model = Sequential()
    model.add(Conv2D(48, kernel_size=3, activation='relu', input_shape=in_data))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(48, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(48, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(96, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(96, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(96, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(384, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(out_data, activation='sigmoid'))

    model.compile(optimizer="adam", loss="MSE", metrics=['AUC'])
    return model


def data_augmentation():
    gen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    return gen


gen = data_augmentation()
callback = [keras.callbacks.ReduceLROnPlateau(patience=1, verbose=1),
            keras.callbacks.ModelCheckpoint(filepath=BEST_MODEL_FILE, monitor='val_loss', verbose=1,
                                            save_best_only=True),
            keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.7 ** x),
            keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)]

model = get_cnn_model(X_train[0].shape, 1)
history = model.fit_generator(gen.flow(X_train, y_train, batch_size=64),
                              epochs=EPOCHS,
                              steps_per_epoch=X_train.shape[0] // 64,
                              validation_data=(X_val, y_val),
                              callbacks=callback,
                              verbose=1)

model = keras.models.load_model(BEST_MODEL_FILE)
results = model.predict(X_test)
utils.save_to_csv(results)