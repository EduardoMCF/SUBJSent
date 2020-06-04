import keras
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input, Convolution1D, MaxPooling1D, Flatten, Concatenate

from .model import Model

class CNN(Model):
    
    def _create(self, input_shape : tuple,
                kernel_sizes : list = [2,3,4],
                num_filters : int = 100,
                activation : str = "relu",
                dense_connections : list = [128],
                dropout_rate : list = [0.5,0.5],
                optimizer : keras.optimizers.Optimizer = Adam,
                learning_rate : float = 0.001,
                loss : str = "binary_crossentropy",
                metrics : list = ['accuracy']) -> keras.models.Model:

        print('inp_shap',input_shape)
        print('kernel_sizes',kernel_sizes)
        print('num_filters', num_filters)
        print('activations', activation)
        print('dense_connectioms', dense_connections)
        print('droput',dropout_rate)
        print('optimiz',optimizer)
        print('loss',loss)
        input_layer = Input(shape = input_shape)
        dropout_layer = Dropout(dropout_rate[0])(input_layer)
        
        filter_maps = []
        for i,size in enumerate(kernel_sizes):
            conv_layer = Convolution1D(filters = num_filters,
                                       kernel_size = size,
                                       activation = activation)(dropout_layer)
            max_pool_layer = MaxPooling1D(pool_size=2)(conv_layer)
            flat_layer = Flatten()(max_pool_layer)
            filter_maps.append(flat_layer)
        filter_maps_conc = Concatenate()(filter_maps)

        dropout_layer = Dropout(dropout_rate[1])(filter_maps_conc)
        dense_layer = Dense(dense_connections[0], activation = activation)(dropout_layer)
        output_layer = Dense(1, activation='sigmoid')(dense_layer)

        model = keras.models.Model(input_layer, output_layer)
        model.compile(loss = loss,
                      optimizer = optimizer(learning_rate = learning_rate),
                      metrics = metrics)

        print(model.summary())
        return model









