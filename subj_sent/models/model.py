from abc import ABC, abstractmethod

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

    def load(self, path):
        self._model = keras.models.load_model(path)

    def save(self, path):
        self._model.save(path)
