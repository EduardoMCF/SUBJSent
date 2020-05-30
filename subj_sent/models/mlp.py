import keras
from keras.layers import Dense, Dropout, Input

from .model import Model

class MLP(Model):
    def __init__(self, input_shape):
        super().__init__(input_shape, **create_params)

    def _create(self, input_shape : int,
                activations : list = ['relu']*3,
                dense_connections : list = [256, 128, 64],
                dropout_rate : list = [0.5,0.5],
                metrics : list = ['accuracy']):

        input_layer = Input(shape=(input_shape,))
        dropout_layer = Dropout(dropout_rate[0])(input_layer)
        dense_layer = Dense(dense_connections[0], activation = activations[0])(dropout_layer)
        dense_layer = Dense(dense_connections[1], activation = activations[1])(dense_layer)
        dense_layer = Dense(dense_connections[2], activation = activations[2])(dense_layer)
        dropout_layer = Dropout(dropout_rate[1])(dense_layer)
        output_layer = Dense(1, activation='sigmoid')(dropout_layer)

        model = keras.models.Model(input_layer, output_layer)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = metrics)

        return model
