import keras
from keras.layers import Dense, Dropout, Input, Convolution1D, MaxPooling1D, Flatten, Concatenate

from .model import Model

class CNN(Model):
    def __init__(self, input_shape, **create_params):
        super().__init__(input_shape, **create_params)
    
    def _create(self, input_shape : tuple = self._input_shape,
                kernel_sizes : list = [2,3,4,7],
                num_filters : int = 100,
                activations : list = ['relu']*2,
                dense_connections : int = 128,
                dropout_rate : list = [0.5,0.5],
                metrics : list = ['accuracy']) -> keras.models.Model:

        input_layer = Input(shape = input_shape)
        dropout_layer = Dropout(dropout_rate[0])(input_layer)
        
        filter_maps = []
        for size in kernel_sizes:
            conv_layer = Convolution1D(filters = num_filters,
                                       kernel_size = size,
                                       activation = activations[0])(dropout_layer)
            max_pool_layer = MaxPooling1D(pool_size=2)(conv_layer)
            flat_layer = Flatten()(max_pool_layer)
            filter_maps.append(flat_layer)
        filter_maps_conc = Concatenate()(filter_maps)

        dropout_layer = Dropout(dropout_rate[1])(filter_maps_conc)
        dense_layer = Dense(dense_connections, activation = activations[1])(dropout_layer)
        output_layer = Dense(1, activation='softmax')(dense_layer)

        model = Model(input_layer, output_layer)
        model.compile(loss="binary_crossentropy", optimizer='adam', metrics=metrics)

        return model









