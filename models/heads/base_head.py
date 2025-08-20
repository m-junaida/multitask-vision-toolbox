from torch import nn
from abc import abstractmethod

class BaseHead(nn.Module):
    """
    Base head for other heads to inherit from.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        """
        Forward function of the head.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, x, data_samples):
        """
        Calculate loss of the head.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x, data_samples):
        """
        Predict results of the head.
        """
        raise NotImplementedError


