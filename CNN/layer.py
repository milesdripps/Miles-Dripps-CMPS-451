"""class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError
"""
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, grad_output, learning_rate):
        pass