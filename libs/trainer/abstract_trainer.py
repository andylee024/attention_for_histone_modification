import abc

class AbstractTrainer(object):
    """Abstract base class for model trainer."""
    __metaclass__ = abc.ABCMeta
   
    @abc.abstractmethod
    def train_model(model, dataset, *args, **kwargs):
        """Train a model on a dataset.

        :param model: model object satisfying abstract model interface
        :param dataset: dataset object satisfying abstract dataset interface
        :return: trained model 
        """
        pass


