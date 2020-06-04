import keras
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input

from .model import Model

class MLP(Model):
    
    def _create(self, input_shape : int,
                activation : str = 'relu',
                dense_connections : list = [256, 128, 64],
                dropout_rate : list = [0.5,0.5],
                optimizer : keras.optimizers.Optimizer = Adam,
                learning_rate : float = 0.001,
                loss : str = "binary_crossentropy",
                metrics : list = ['accuracy']) -> keras.models.Model:

        input_layer = Input(shape=input_shape)
        dropout_layer = Dropout(dropout_rate[0])(input_layer)
        dense_layer = Dense(dense_connections[0], activation = activation)(dropout_layer)
        dense_layer = Dense(dense_connections[1], activation = activation)(dense_layer)
        dense_layer = Dense(dense_connections[2], activation = activation)(dense_layer)
        dropout_layer = Dropout(dropout_rate[1])(dense_layer)
        output_layer = Dense(1, activation='sigmoid')(dropout_layer)

        model = keras.models.Model(input_layer, output_layer)
        model.compile(loss = loss,
                      optimizer = optimizer(learning_rate = learning_rate),
                      metrics = metrics)

        return model
