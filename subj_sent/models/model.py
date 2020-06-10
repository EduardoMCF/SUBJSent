from abc import ABC, abstractmethod
import tensorflow.keras as keras

class Model(ABC):
    def __init__(self, input_shape, **create_params):
        self._input_shape = input_shape
        self._model = self._create(self._input_shape, **create_params)

    @abstractmethod
    def _create(self, **params) -> keras.models.Model:
        raise NotImplementedError

    def train(self, X, y, **params) -> keras.callbacks.History:
        return self._model.fit(X, y, **params)

    def evaluate(self, X, y) -> list:
        return self._model.evaluate(X,y)

    def predict(self, data) -> list:
        return self._model.predict(data)

    @classmethod
    def load(cls, path):
        loaded_model = keras.models.load_model(path)
        input_shape = loaded_model.input_shape[1:]
        model = cls(input_shape)
        model._model = loaded_model
        return model

    def save(self, path):
        self._model.save(path)
