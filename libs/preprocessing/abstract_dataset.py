import abc

class AbstractDataset(object):
    """Abstract base class for a dataset object containing training examples."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_training_example(self, index):
        """Get single training example.
        
        :param index: query index for training example.
        :return: training example corresponding to query index.
        """
        pass

    @abc.abstractmethod
    def get_training_examples(self, indices):
        """Get a list of training examples.

        :param indices: query indices for training examples (list or numpy array).
        :return: list of 2-tuples of the form (idx, training_example).
        """
        pass

