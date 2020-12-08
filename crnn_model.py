import utils as utils
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import ReLU
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import config
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import keras

BEST_MODEL_FILE = "./models/best_crnn_model_trained.hdf5"
EPOCHS = 75
BATCH_SIZE = 300

X, Y = utils.load_training_data()
X_test = utils.load_testing_data()
if config.options()["PSEUDO_LABELLING"]:
    x_pseudo_label, y_pseudo_label = utils.load_pseudo_test_data()
    X = np.concatenate(X, x_pseudo_label)
    Y = np.concatenate(Y, y_pseudo_label)
X = X.reshape(-1, 500, 40)
X_test = X_test.reshape(-1, 500, 40)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


class Attention(Layer):
    # Class taken from https://www.kaggle.com/kbhartiya83/sound-recogniser-bidirectionallstm-attention
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def get_rcnn_model(in_data, out_data):
    model = Sequential()
    model.add(BatchNormalization(input_shape=in_data))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(ReLU())
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(ReLU())
    model.add(Dropout(0.4))

    model.add(Attention(500))
    model.add(Dense(out_data, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


callback = [keras.callbacks.ReduceLROnPlateau(patience=1, verbose=1),
            keras.callbacks.ModelCheckpoint(filepath=BEST_MODEL_FILE, monitor='val_loss', verbose=1,
                                            save_best_only=True),
            keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.7 ** x),
            keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)]

model = get_rcnn_model(X_train[0].shape, 1)
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(X_val, y_val),
          callback=callback)

model = keras.models.load_model(BEST_MODEL_FILE)
results = model.predict(X_test)
utils.save_to_csv(results)
