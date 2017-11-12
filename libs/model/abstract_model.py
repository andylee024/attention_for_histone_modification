import abc

class AbstractModel(object):
    """Abstract base class for machine learning models."""
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def predict(self, *args):
        """Compute model prediction.

        :param *args : input data X specific to model.
        :return y : predictions based on input data.
        """
        pass


class AbstractTensorflowModel(AbstractModel):
    """Abstract base class specific to tensorflow models."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict(self, *args):
        """Compute model prediction.

        :param *args : input data X specific to model.
        :return y : predictions based on input data.
        """
        pass

    @abc.abstractproperty
    def inputs(self):
        """Return placeholder objects corresponding to tensorflow model inputs.

        Tensorflow models operate on tf.placeholder objects that represent inputs to
        the model. This needs to be exposed as a public interface so that those placeholders
        can be populated.
        """
        pass

    @abc.abstractproperty
    def outputs(self):
        """Return placeholder objects corresponding to tensorflow model outputs."""
        pass
