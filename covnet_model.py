from keras.models import Sequential
from keras.layers.core import Dropout, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv1D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import utils
import keras
import config
from keras.preprocessing.image import ImageDataGenerator


BEST_MODEL_FILE = "./models/best_covnet_model_trained.hdf5"
EPOCHS = 50
FILTER = 40

X, Y = utils.load_training_data()
X_test = utils.load_testing_data()
if config.options()["PSEUDO_LABELLING"]:
    x_pseudo_label, y_pseudo_label = utils.load_pseudo_test_data()
    X = np.concatenate(X, x_pseudo_label)
    Y = np.concatenate(Y, y_pseudo_label)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


def get_covnet_model(in_data, out_data):
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)
    model = Sequential()

    # conv1
    model.add(ZeroPadding2D((2, 2), input_shape=in_data))
    model.add(Conv2D(FILTER, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    # learnedpool1
    model.add(Conv2D(FILTER, (5, 1), strides=(5, 1), activation='relu'))
    model.add(Dropout(0.50))

    # conv2
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(FILTER, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    # learnedpool2
    model.add(Conv2D(FILTER, (2, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.50))

    # conv3
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(FILTER, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    # learnedpool3
    model.add(Conv2D(FILTER, (2, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.50))

    # conv4
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(FILTER, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    # learnedpool4
    model.add(Conv2D(FILTER, (2, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.50))

    # stacking(reshaping)
    model.add(Reshape((500, FILTER)))

    # learnedpool5
    model.add(Conv1D(FILTER, (500), strides=(1), activation='relu'))
    model.add(Dropout(0.50))

    # fully connected layers using conv
    model.add(Reshape((1, 1, FILTER)))

    model.add(Conv2D(196, (1, 1), activation='softmax'))
    model.add(Dropout(0.50))

    model.add(Conv2D(196, (1, 1), activation='softmax'))
    model.add(Dropout(0.50))

    model.add(Conv2D(out_data, (1, 1), activation='sigmoid'))
    model.add(Reshape((out_data,)))
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

model = get_covnet_model(X_train[0].shape, 1)
history = model.fit_generator(gen.flow(X_train, y_train, batch_size=64),
                              epochs=EPOCHS,
                              steps_per_epoch=X_train.shape[0] // 64,
                              validation_data=(X_val, y_val),
                              callbacks=callback,
                              verbose=1)

model = keras.models.load_model(BEST_MODEL_FILE)
results = model.predict(X_test)
utils.save_to_csv(results)