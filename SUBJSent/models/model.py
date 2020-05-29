from abc import ABC, abstractmethod


class Model(ABC):

	@abstractmethod
    def create(self, **params):
    	raise NotImplementedError

    @abstractmethod
    def train(self, x_train, y_train, **params):
        raise NotImplementedError

    @abstractmethod
    def eval(self, data):
    	raise NotImplementedError

    @abstractmethod
    def predict(self, x_data, y_data):
    	raise NotImplementedError

    @abstractmethod
   	def load(self, path):
    	raise NotImplementedError

    @abstractmethod
    def save(self, path):
    	raise NotImplementedError
