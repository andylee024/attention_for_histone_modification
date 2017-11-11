import abc

class abstract_model(object):
    """Abstract base class for neural network model."""
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def predict(self):
        """Predict based on neural network input."""
        pass

    @abc.abstractmethod
    def get_model_inputs(self):
        """Return dictionary of tf.placeholder objects representing model inputs."""

